from typing import Tuple, Dict

import torch
from Model_Training.pipelines.base_stage import BaseStage


class DistanceNormalizationStage(BaseStage):
    """
    Normalizes face landmarks to be invariant to distance from the camera.
    Assumes input tensor is of shape (num_frames, num_landmarks, num_coordinates).
    Normalization is applied frame-wise.
    """
    def __init__(self, nose_tip_index: int = 1, left_eye_outer_corner_index: int = 33,
                 right_eye_outer_corner_index: int = 263, verbose: bool = False, **kwargs):
        """
        Initializes the normalization stage.

        Args:
            nose_tip_index (int): Index of the nose tip landmark.
                                  IMPORTANT: Verify this for your 478 landmark model.
            left_eye_outer_corner_index (int): Index of the left eye outer corner landmark.
                                               IMPORTANT: Verify this for your 478 landmark model.
            right_eye_outer_corner_index (int): Index of the right eye outer corner landmark.
                                                IMPORTANT: Verify this for your 478 landmark model.
            verbose (bool): If True, prints processing messages.
        """
        super().__init__(verbose, **kwargs)
        self.nose_tip_index = nose_tip_index
        self.left_eye_outer_corner_index = left_eye_outer_corner_index
        self.right_eye_outer_corner_index = right_eye_outer_corner_index
        self.verbose_stage = verbose # Renamed to avoid conflict with process method's verbose


    def process(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies distance normalization to the landmark tensor.

        Args:
            x (torch.Tensor): Landmark tensor of shape (T, N, C),
                                 where T is frames, N is landmarks, C is coordinates (x,y,z).
            y (Dict[str, torch.Tensor]): Dictionary of labels (not modified in this stage).

        Returns:
            torch.Tensor: Normalized landmark tensor of the same shape.
        """
        if not isinstance(x, torch.Tensor):
            if self.verbose_stage:
                print("DistanceNormalizationStage: Input is not a tensor. Skipping.")
            return x, y
        if x.ndim != 3 or x.shape[1] <= max(self.nose_tip_index, self.left_eye_outer_corner_index, self.right_eye_outer_corner_index):
            if self.verbose_stage:
                print(f"DistanceNormalizationStage: Invalid tensor shape {x.shape} or insufficient landmarks. Skipping.")
            return x, y

        if self.verbose_stage:
            print(f"DistanceNormalizationStage: Processing tensor of shape {x.shape}")

        normalized_frames = []
        for frame_idx in range(x.shape[0]):
            frame_landmarks = x[frame_idx, :, :] # Shape (N, C)

            # 1. Select Reference Landmarks
            try:
                center_landmark_coords = frame_landmarks[self.nose_tip_index, :] # (C,)
                p1_coords = frame_landmarks[self.left_eye_outer_corner_index, :]   # (C,)
                p2_coords = frame_landmarks[self.right_eye_outer_corner_index, :]   # (C,)
            except IndexError:
                if self.verbose_stage:
                    print(f"DistanceNormalizationStage: Landmark index out of bounds for frame {frame_idx}. Skipping frame.")
                normalized_frames.append(frame_landmarks) # Append original if error
                continue

            # 2a. Translate to the Chosen Origin (nose tip)
            # Ensure broadcasting if C > 1, though typically C=3
            translated_landmarks = frame_landmarks - center_landmark_coords.unsqueeze(0) # (N, C) - (1, C) -> (N, C)

            # 2b. Calculate Scale Factor (using X and Y coordinates for stability, as Z can be noisy)
            # Using only X and Y for scale_distance calculation
            scale_distance = torch.sqrt(
                (p1_coords[0] - p2_coords[0])**2 + \
                (p1_coords[1] - p2_coords[1])**2
            )

            if scale_distance < 1e-6: # Avoid division by zero or very small numbers
                if self.verbose_stage:
                    print(f"DistanceNormalizationStage: Scale distance is near zero for frame {frame_idx}. Skipping scaling for this frame.")
                normalized_frames.append(translated_landmarks) # Append translated but not scaled
                continue

            # 2c. Scale the Translated Landmarks
            # We scale all coordinates (x, y, z) by this 2D-derived scale factor
            scaled_landmarks = translated_landmarks / scale_distance
            normalized_frames.append(scaled_landmarks)

        if not normalized_frames: # Should not happen if input x was valid
             return x, y

        output_tensor = torch.stack(normalized_frames, dim=0)
        if self.verbose_stage:
            print(f"DistanceNormalizationStage: Finished processing. Output shape: {output_tensor.shape}")
        return output_tensor, y
