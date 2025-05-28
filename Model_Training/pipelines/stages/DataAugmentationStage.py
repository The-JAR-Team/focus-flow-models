import math
import random
from typing import Optional, Dict, Tuple

import torch

from Model_Training.pipelines.base_stage import BaseStage


class DataAugmentationStage(BaseStage):
    """
    Applies various augmentations to landmark data.
    Assumes input tensor is of shape (num_frames, num_landmarks, num_coordinates).
    Augmentations are applied frame-wise or over short temporal bursts.
    """

    def __init__(self,
                 add_noise_prob: float = 0.0,
                 noise_std: float = 0.01,
                 random_scale_prob: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 random_rotate_prob: float = 0.0,
                 max_rotation_angle_deg: float = 15.0,
                 random_flip_prob: float = 0.0,
                 landmark_flip_map: Optional[Dict[int, int]] = None,
                 temporal_jitter_prob: float = 0.0,
                 jitter_burst_length_range: Tuple[int, int] = (2, 5),  # Default, can be overridden by config
                 jitter_magnitude_std: float = 0.03,
                 max_jitter_bursts_per_sequence: int = 1,  # New parameter
                 verbose: bool = False,
                 **kwargs):
        """
        Initializes the data augmentation stage.

        Args:
            add_noise_prob (float): Probability of adding Gaussian noise.
            noise_std (float): Standard deviation of the Gaussian noise.
            random_scale_prob (float): Probability of applying random scaling.
            scale_range (Tuple[float, float]): Min and max scale factor.
            random_rotate_prob (float): Probability of applying random 2D rotation.
            max_rotation_angle_deg (float): Maximum rotation angle in degrees.
            random_flip_prob (float): Probability of applying horizontal flip.
            landmark_flip_map (Optional[Dict[int, int]]): Mapping for horizontal flip.
            temporal_jitter_prob (float): Probability of applying a single temporal
                                          displacement jitter burst. This check is done for
                                          each potential burst up to max_jitter_bursts_per_sequence.
            jitter_burst_length_range (Tuple[int, int]): Range (min_frames, max_frames) for
                                                         the duration of a jitter burst.
            jitter_magnitude_std (float): Standard deviation for the random displacement vector.
            max_jitter_bursts_per_sequence (int): Maximum number of jitter bursts that can be
                                                  applied to a single sequence.
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

        self.temporal_jitter_prob = temporal_jitter_prob
        if not (isinstance(jitter_burst_length_range, tuple) and len(jitter_burst_length_range) == 2 and
                jitter_burst_length_range[0] <= jitter_burst_length_range[1] and jitter_burst_length_range[0] > 0):
            raise ValueError(
                "jitter_burst_length_range must be a tuple (min_frames, max_frames) with min_frames > 0 and min_frames <= max_frames.")
        self.jitter_burst_length_range = jitter_burst_length_range
        self.jitter_magnitude_std = jitter_magnitude_std
        if not (isinstance(max_jitter_bursts_per_sequence, int) and max_jitter_bursts_per_sequence >= 0):
            raise ValueError("max_jitter_bursts_per_sequence must be a non-negative integer.")
        self.max_jitter_bursts_per_sequence = max_jitter_bursts_per_sequence

        self.verbose_stage = verbose

        if self.random_flip_prob > 0 and self.landmark_flip_map is None:
            raise ValueError("landmark_flip_map is required when random_flip_prob > 0.")

    def process(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies augmentations to the landmark tensor.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            if self.verbose_stage: print(f"{self.__class__.__name__}: Input x is not a valid 3D tensor. Skipping.")
            return x, y

        if x.shape[0] == 0:
            if self.verbose_stage: print(f"{self.__class__.__name__}: Input x has 0 frames. Skipping.")
            return x, y

        num_total_frames, num_landmarks, num_coords = x.shape
        x_augmented = x.clone()

        # --- Temporal Displacement Jitter ---
        # Apply up to max_jitter_bursts_per_sequence, each with its own probability check
        jitter_bursts_applied_count = 0
        for _ in range(self.max_jitter_bursts_per_sequence):
            if random.random() < self.temporal_jitter_prob and num_total_frames >= self.jitter_burst_length_range[0]:
                # Determine burst length for this specific burst
                # Ensure burst_len does not exceed total frames
                max_possible_burst_len = min(self.jitter_burst_length_range[1], num_total_frames)
                if self.jitter_burst_length_range[0] > max_possible_burst_len:  # Min requested > max possible
                    if self.verbose_stage: print(
                        f"{self.__class__.__name__}: Skipping a temporal jitter burst, sequence too short for min_burst_length.")
                    continue  # Skip this potential burst

                burst_len = random.randint(self.jitter_burst_length_range[0], max_possible_burst_len)

                # Determine start frame for this burst
                # Ensure the burst fits: start_frame_idx can go from 0 to num_total_frames - burst_len
                if num_total_frames - burst_len < 0:  # Should be caught by above check, but as safeguard
                    if self.verbose_stage: print(
                        f"{self.__class__.__name__}: Skipping a temporal jitter burst, sequence too short for chosen burst_len {burst_len}.")
                    continue

                start_frame_idx = random.randint(0, num_total_frames - burst_len)

                if self.verbose_stage:
                    print(
                        f"{self.__class__.__name__}: Applying temporal jitter burst #{jitter_bursts_applied_count + 1} for {burst_len} frames starting at {start_frame_idx}.")

                for i in range(burst_len):
                    frame_to_jitter_idx = start_frame_idx + i
                    displacement_vector = torch.randn(num_coords, device=x_augmented.device) * self.jitter_magnitude_std
                    x_augmented[frame_to_jitter_idx, :, :] += displacement_vector.unsqueeze(0)

                jitter_bursts_applied_count += 1
            elif num_total_frames < self.jitter_burst_length_range[
                0] and random.random() < self.temporal_jitter_prob:  # Log if prob passed but too short
                if self.verbose_stage: print(
                    f"{self.__class__.__name__}: Wanted to apply temporal jitter, but sequence length ({num_total_frames}) is less than min burst length ({self.jitter_burst_length_range[0]}).")

        # --- Frame-wise Augmentations (applied after potential temporal jitter) ---
        # These are applied to the `x_augmented` tensor which might have been modified by jitter.
        processed_frames_for_static_augs = []
        for frame_idx in range(x_augmented.shape[0]):
            current_landmarks = x_augmented[frame_idx, :, :].clone()

            if random.random() < self.random_flip_prob and self.landmark_flip_map:
                flipped_landmarks = current_landmarks.clone()
                flipped_landmarks[:, 0] = -flipped_landmarks[:, 0]
                temp_landmarks = flipped_landmarks.clone()
                for i in range(num_landmarks):
                    target_idx = self.landmark_flip_map.get(i)
                    if target_idx is not None and target_idx != i:
                        temp_landmarks[i, :] = flipped_landmarks[target_idx, :]
                current_landmarks = temp_landmarks

            centroid = torch.mean(current_landmarks[:, :min(2, num_coords)], dim=0)

            if random.random() < self.random_rotate_prob and num_coords >= 2:
                angle = random.uniform(-self.max_rotation_angle_rad, self.max_rotation_angle_rad)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                xy_coords = current_landmarks[:, :2] - centroid.unsqueeze(0)
                x_rot = xy_coords[:, 0] * cos_a - xy_coords[:, 1] * sin_a
                y_rot = xy_coords[:, 0] * sin_a + xy_coords[:, 1] * cos_a
                current_landmarks[:, 0] = x_rot + centroid[0]
                current_landmarks[:, 1] = y_rot + centroid[1]

            if random.random() < self.random_scale_prob:
                scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
                scaled_coords = current_landmarks.clone()
                if num_coords >= 2:
                    scaled_coords[:, :2] = (current_landmarks[:, :2] - centroid.unsqueeze(
                        0)) * scale_factor + centroid.unsqueeze(0)
                if num_coords > 2:
                    z_centroid = torch.mean(current_landmarks[:, 2])
                    scaled_coords[:, 2] = (current_landmarks[:, 2] - z_centroid) * scale_factor + z_centroid
                current_landmarks = scaled_coords

            if random.random() < self.add_noise_prob:
                noise = torch.randn_like(current_landmarks) * self.noise_std
                current_landmarks += noise

            processed_frames_for_static_augs.append(current_landmarks)

        if processed_frames_for_static_augs:
            x_augmented = torch.stack(processed_frames_for_static_augs, dim=0)

        return x_augmented, y
