import math
import random
from typing import Optional, Dict, Tuple

import torch

from Model_Training.pipelines.base_stage import BaseStage


class DataAugmentationStage(BaseStage):
    """
    Applies various augmentations to landmark data.
    Assumes input tensor is of shape (num_frames, num_landmarks, num_coordinates).
    Augmentations are applied frame-wise.
    """

    def __init__(self, add_noise_prob: float = 0.0, noise_std: float = 0.01, random_scale_prob: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1), random_rotate_prob: float = 0.0,
                 max_rotation_angle_deg: float = 15.0, random_flip_prob: float = 0.0,
                 landmark_flip_map: Optional[Dict[int, int]] = None, verbose: bool = False, **kwargs):
        """
        Initializes the data augmentation stage.

        Args:
            add_noise_prob (float): Probability of adding Gaussian noise.
            noise_std (float): Standard deviation of the Gaussian noise.
            random_scale_prob (float): Probability of applying random scaling.
            scale_range (Tuple[float, float]): Min and max scale factor.
            random_rotate_prob (float): Probability of applying random 2D rotation (around Z-axis).
            max_rotation_angle_deg (float): Maximum rotation angle in degrees.
            random_flip_prob (float): Probability of applying horizontal flip.
            landmark_flip_map (Optional[Dict[int, int]]): A dictionary mapping a landmark index
                to its horizontally flipped counterpart. Required if random_flip_prob > 0.
                The map should contain entries for all landmarks that change position during a flip.
                Symmetric landmarks (e.g., on the nose bridge midline) might map to themselves or be omitted.
            verbose (bool): If True, prints processing messages.
        """
        super().__init__(verbose, **kwargs)
        self.add_noise_prob = add_noise_prob
        self.noise_std = noise_std
        self.random_scale_prob = random_scale_prob
        self.scale_range = scale_range
        self.random_rotate_prob = random_rotate_prob
        self.max_rotation_angle_rad = math.radians(max_rotation_angle_deg)
        self.random_flip_prob = random_flip_prob
        self.landmark_flip_map = landmark_flip_map
        self.verbose_stage = verbose

        if self.random_flip_prob > 0 and self.landmark_flip_map is None:
            raise ValueError("landmark_flip_map is required when random_flip_prob > 0.")

    def process(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies augmentations to the landmark tensor.

        Args:
            x (torch.Tensor): Input landmark tensor of shape (num_frames, num_landmarks, num_coordinates).
            y (Dict[str, torch.Tensor]): Dictionary of labels (not used in this stage).
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Augmented landmark tensor and unchanged label dictionary.
        """
        if not isinstance(x, torch.Tensor):
            return x, y
        if x.ndim != 3 or x.shape[2] < 2:  # Need at least X, Y
            return x, y


        augmented_frames = []
        num_landmarks = x.shape[1]

        for frame_idx in range(x.shape[0]):
            current_landmarks = x[frame_idx, :, :].clone()  # Shape (N, C)

            # --- Horizontal Flip ---
            if random.random() < self.random_flip_prob and self.landmark_flip_map:
                flipped_landmarks = current_landmarks.clone()
                # Negate X coordinates (assuming X is the horizontal axis, index 0)
                flipped_landmarks[:, 0] = -flipped_landmarks[:, 0]

                # Create a temporary tensor to hold the re-ordered landmarks
                temp_landmarks = flipped_landmarks.clone()
                for i in range(num_landmarks):
                    # If landmark i has a mapping, place the flipped data of its counterpart into position i
                    # If landmark i is its own counterpart (e.g. midline), its data is already flipped (X negated)
                    # If landmark i is not in map (e.g. midline and not explicitly mapped to itself), it keeps its X-negated data
                    target_idx = self.landmark_flip_map.get(i)
                    if target_idx is not None and target_idx != i:  # if i maps to a different landmark
                        temp_landmarks[i, :] = flipped_landmarks[target_idx, :]
                    # If i maps to itself (explicitly in map) or not in map at all,
                    # flipped_landmarks[i,:] (with X negated) is already correct for temp_landmarks[i,:]
                current_landmarks = temp_landmarks

            # Calculate centroid for scale and rotation (using X, Y coordinates: indices 0, 1)
            # This is important if data isn't already centered (e.g. if DistanceNormalizationStage wasn't used)
            # or if previous augmentations shifted the centroid.
            centroid = torch.mean(current_landmarks[:, :2], dim=0)  # Shape (2,) for (x_mean, y_mean)

            # --- Random Rotation (2D in-plane around Z-axis) ---
            if random.random() < self.random_rotate_prob:
                angle = random.uniform(-self.max_rotation_angle_rad, self.max_rotation_angle_rad)
                cos_a, sin_a = math.cos(angle), math.sin(angle)

                # Rotation matrix for 2D (XY plane)
                # R = [[cos_a, -sin_a],
                #      [sin_a,  cos_a]]

                # Center XY coordinates around centroid
                xy_coords = current_landmarks[:, :2] - centroid.unsqueeze(0)  # (N, 2)

                # Apply rotation
                x_rot = xy_coords[:, 0] * cos_a - xy_coords[:, 1] * sin_a
                y_rot = xy_coords[:, 0] * sin_a + xy_coords[:, 1] * cos_a

                # Update XY coordinates and shift back
                current_landmarks[:, 0] = x_rot + centroid[0]
                current_landmarks[:, 1] = y_rot + centroid[1]
                # Z coordinate (if present, index 2) remains unchanged by 2D rotation

            # --- Random Scaling ---
            if random.random() < self.random_scale_prob:
                scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

                # Center all coordinates around their respective means for scaling
                # For X, Y, use the previously calculated centroid
                # For Z (if present), calculate its mean separately
                scaled_coords = current_landmarks.clone()
                scaled_coords[:, :2] = (current_landmarks[:, :2] - centroid.unsqueeze(
                    0)) * scale_factor + centroid.unsqueeze(0)
                if current_landmarks.shape[1] > 2:  # If Z coordinate exists
                    z_centroid = torch.mean(current_landmarks[:, 2])
                    scaled_coords[:, 2] = (current_landmarks[:, 2] - z_centroid) * scale_factor + z_centroid
                current_landmarks = scaled_coords

            # --- Add Noise ---
            if random.random() < self.add_noise_prob:
                noise = torch.randn_like(current_landmarks) * self.noise_std
                current_landmarks += noise

            augmented_frames.append(current_landmarks)

        if not augmented_frames:
            return x, y

        output_tensor = torch.stack(augmented_frames, dim=0)

        return output_tensor, y
