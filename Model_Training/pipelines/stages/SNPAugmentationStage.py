import math
import random
from typing import Dict, Tuple, Mapping  # Mapping was in your original imports, kept for consistency

import torch
from Model_Training.pipelines.base_stage import BaseStage


class SNPAugmentationStage(BaseStage):
    """
    Augments data by converting a percentage of samples to 'Subject Not Present' (SNP).
    If a sample is chosen for SNP conversion, a random portion of its frames (above a minimum threshold)
    will have their landmark data set to -1, and its labels will be updated to reflect SNP.
    """

    def __init__(self,
                 snp_conversion_prob: float,
                 min_snp_frame_percentage: float,
                 snp_class_idx: int,
                 snp_reg_score: float,
                 verbose: bool = False,
                 **kwargs):
        """
        Initializes the SNP augmentation stage.

        Args:
            snp_conversion_prob (float): Probability that a given sample will be converted to SNP.
                                         Value should be between 0.0 and 1.0.
            min_snp_frame_percentage (float): Minimum percentage of frames to set to -1 if a sample
                                              is chosen for SNP conversion. Value should be
                                              between 0.0 and 1.0. The actual percentage
                                              will be randomly chosen between this value and 1.0.
            snp_class_idx (int): The classification index to assign to samples converted to SNP.
            snp_reg_score (float): The regression score to assign to samples converted to SNP.
            verbose (bool): If True, prints processing messages.
        """
        super().__init__(verbose, **kwargs)  # Pass verbose to BaseStage
        if not (0.0 <= snp_conversion_prob <= 1.0):
            raise ValueError("snp_conversion_prob must be between 0.0 and 1.0")
        if not (0.0 <= min_snp_frame_percentage <= 1.0):
            raise ValueError("min_snp_frame_percentage must be between 0.0 and 1.0")

        self.snp_conversion_prob = snp_conversion_prob
        self.min_snp_frame_percentage = min_snp_frame_percentage
        self.snp_class_idx = snp_class_idx
        self.snp_reg_score = snp_reg_score
        # self.verbose is inherited from BaseStage and set by super().__init__

    def process(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies SNP augmentation to the landmark tensor and updates labels accordingly.
        This method has a single return point.

        Args:
            x (torch.Tensor): Input landmark tensor of shape (num_frames, num_landmarks, num_coordinates).
            y (Dict[str, torch.Tensor]): Dictionary of labels, expected to contain
                                         'classification_targets' and 'regression_targets'.
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Potentially augmented landmark tensor
                                                         and potentially updated label dictionary.
        """
        # Initialize return variables with the input values
        x_processed = x
        y_processed = y

        # Flag to indicate if processing should proceed
        can_process = True

        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            if self.verbose:
                print(
                    f"{self.__class__.__name__}: Input x is not a valid 3D tensor (shape {x.shape if isinstance(x, torch.Tensor) else type(x)}). Skipping.")
            can_process = False

        if can_process and (
                not isinstance(y, dict) or 'classification_targets' not in y or 'regression_targets' not in y):
            if self.verbose:
                print(f"{self.__class__.__name__}: Input y is not a valid dictionary with required keys. Skipping.")
            can_process = False

        if can_process and random.random() < self.snp_conversion_prob:
            num_frames = x.shape[0]
            if num_frames == 0:
                if self.verbose:
                    print(f"{self.__class__.__name__}: Input x has 0 frames. Skipping SNP conversion for this sample.")
                can_process = False  # Cannot process if no frames

            if can_process:  # Check again if num_frames was > 0
                # Determine the percentage of frames to convert to SNP for this sample
                actual_snp_frame_percentage = random.uniform(self.min_snp_frame_percentage, 1.0)
                # Use math.ceil to ensure at least one frame is modified if actual_snp_frame_percentage > 0
                num_frames_to_snp = math.ceil(actual_snp_frame_percentage * num_frames)

                if num_frames_to_snp > 0:
                    # Clone x to avoid modifying the original tensor that might be used elsewhere.
                    x_modified = x.clone()  # Use x, not x_processed, as x_processed might be a reference

                    frame_indices_all = list(range(num_frames))
                    random.shuffle(frame_indices_all)
                    # These are Python list of ints, PyTorch handles this for indexing
                    snp_selected_frame_indices = frame_indices_all[:num_frames_to_snp]

                    if self.verbose:
                        print(
                            f"{self.__class__.__name__}: Converting sample to SNP. Modifying {num_frames_to_snp}/{num_frames} frames to -1.")

                    # This sets all landmarks and all coordinates for the selected frames to -1.0
                    x_modified[snp_selected_frame_indices, :, :] = -1.0
                    x_processed = x_modified  # Update the return variable

                    # Update labels to SNP
                    # Ensure new label tensors are on the same device as original y tensors
                    original_class_device = y['classification_targets'].device
                    original_reg_device = y['regression_targets'].device

                    # Create a new dictionary for the updated labels
                    y_updated = y.copy()  # Start by copying all existing label key-values
                    y_updated['classification_targets'] = torch.tensor(self.snp_class_idx, dtype=torch.long,
                                                                       device=original_class_device)
                    y_updated['regression_targets'] = torch.tensor(self.snp_reg_score, dtype=torch.float32,
                                                                   device=original_reg_device)

                    y_processed = y_updated  # Update the return variable
                    #printing resulting x and y after SNP augmentation
                    print(f"{x_processed}")
                    print(f"{y_processed}")

        return x_processed, y_processed
