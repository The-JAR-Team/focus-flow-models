from typing import List

import torch
from Preprocess.Pipeline.PipelineStage import PipelineStage


class FeatureSelectorStage(PipelineStage):
    """
    Selects a subset of landmarks (features).
    Assumes input tensor is of shape (num_frames, num_landmarks, num_coordinates).
    """
    def __init__(self, landmark_indices_to_keep: List[int], verbose: bool = False):
        """
        Initializes the feature selection stage.

        Args:
            landmark_indices_to_keep (List[int]): A list of landmark indices to select.
            verbose (bool): If True, prints processing messages.
        """
        self.indices = sorted(list(set(landmark_indices_to_keep))) # Ensure unique and sorted
        self.verbose_stage = verbose
        if self.verbose_stage:
            print(f"FeatureSelectorStage: Initialized to keep indices: {self.indices}")

    def process(self, data: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """
        Applies feature selection to the landmark tensor.

        Args:
            data (torch.Tensor): Landmark tensor of shape (T, N, C).
            verbose (bool): If True, prints detailed status messages.

        Returns:
            torch.Tensor: Landmark tensor of shape (T, num_selected_landmarks, C).
        """
        if not isinstance(data, torch.Tensor):
            if self.verbose_stage or verbose:
                print("FeatureSelectorStage: Input is not a tensor. Skipping.")
            return data
        if data.ndim != 3:
            if self.verbose_stage or verbose:
                print(f"FeatureSelectorStage: Invalid tensor shape {data.shape}. Expected 3D. Skipping.")
            return data

        if self.verbose_stage or verbose:
            print(f"FeatureSelectorStage: Processing tensor of shape {data.shape}. Selecting {len(self.indices)} landmarks.")

        try:
            selected_data = data[:, self.indices, :]
        except IndexError:
            max_index_requested = max(self.indices) if self.indices else -1
            if self.verbose_stage or verbose:
                print(f"FeatureSelectorStage: Error selecting indices. Max index requested: {max_index_requested}, "
                      f"but tensor has {data.shape[1]} landmarks. Returning original data.")
            return data # Return original data if selection fails

        if self.verbose_stage or verbose:
            print(f"FeatureSelectorStage: Finished processing. Output shape: {selected_data.shape}")
        return selected_data
