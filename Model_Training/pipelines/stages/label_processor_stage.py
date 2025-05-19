import torch
from typing import Tuple, Dict, Any, Mapping

from Model_Training.pipelines.base_stage import BaseStage


class LabelProcessorStage(BaseStage):
    def __init__(self,
                 label_to_idx_map: Mapping[str, int],
                 idx_to_score_map: Mapping[int, float],
                 engagement_key: str = 'engagement_string',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(verbose, **kwargs)
        self.label_to_idx_map = label_to_idx_map
        self.idx_to_score_map = idx_to_score_map
        self.engagement_key = engagement_key  # Key in the raw label dict that holds the string label

    def process(self, x: torch.Tensor, y_raw: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes the raw label dictionary (y_raw) to create multi-task labels.
        'x' (tensor_stack) is passed through unchanged.

        Args:
            x (torch.Tensor): The input tensor_stack.
            y_raw (Dict[str, Any]): The raw label dictionary loaded from the .pt file.
                                     Expected to contain a key like 'engagement_string'.
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - The original x tensor.
                - A new y dictionary: {'regression_targets': tensor, 'classification_targets': tensor}
                  Returns (x, {}) or raises an error if processing fails.
        """
        if not isinstance(y_raw, dict):
            if self.verbose:
                print(f"LabelProcessorStage: y_raw is not a dict (got {type(y_raw)}). Cannot process labels.")
            # Or raise ValueError("y_raw must be a dictionary for LabelProcessorStage")
            return x, {}  # Return empty dict for y to signal an issue or allow filtering

        engagement_string_value = y_raw.get(self.engagement_key)
        if engagement_string_value is None:
            if self.verbose:
                print(f"LabelProcessorStage: Key '{self.engagement_key}' not found in y_raw. Cannot process labels.")
            return x, {}

        processed_lbl_str = str(engagement_string_value).strip()
        class_idx = self.label_to_idx_map.get(processed_lbl_str)

        if class_idx is None:
            if self.verbose:
                print(
                    f"LabelProcessorStage: Unknown engagement string '{processed_lbl_str}'. Cannot map to class index.")
            return x, {}

        regression_score = self.idx_to_score_map.get(class_idx)

        if regression_score is None:
            if self.verbose:
                print(
                    f"LabelProcessorStage: Score not found for class_idx {class_idx}. Cannot create regression target.")
            return x, {}

        multi_task_labels = {
            'regression_targets': torch.tensor(regression_score, dtype=torch.float32),
            'classification_targets': torch.tensor(class_idx, dtype=torch.long)
        }

        if self.verbose:
            print(
                f"  LabelProcessorStage: Processed '{processed_lbl_str}' -> Class {class_idx}, Score {regression_score:.2f}")

        return x, multi_task_labels  # x is unchanged, y is now the multi_task_labels dict