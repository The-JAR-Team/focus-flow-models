import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Any, Dict, Mapping  # Mapping for type hint

from Model_Training.pipelines.pipeline import OrchestrationPipeline


class CachedTensorDataset(Dataset):
    def __init__(self,
                 cache_dir: str,
                 transform_pipeline: Optional[OrchestrationPipeline] = None,
                 verbose_dataset: bool = False):
        """
        Recursively collects all .pt files in cache_dir.
        Assumes each file is a dict with "tensor_stack" and "label" (raw label dict).
        The transform_pipeline is expected to handle all processing, including
        transforming the raw label dict into the multi-task label format.

        Args:
            cache_dir (str): Directory containing the cached .pt files.
            transform_pipeline (Optional[OrchestrationPipeline]): An OrchestrationPipeline
                instance to process the 'tensor_stack' (X) and 'label' (Y_raw)
                after loading. The first stage of this pipeline should ideally be
                LabelProcessorStage.
            verbose_dataset (bool): If True, dataset-level messages are printed.
        """
        self.cache_dir = cache_dir
        self.files = []
        self.transform_pipeline = transform_pipeline
        self.verbose_dataset = verbose_dataset

        if not os.path.isdir(cache_dir):
            if self.verbose_dataset:
                print(f"CachedTensorDataset: Cache directory '{cache_dir}' not found. Dataset will be empty.")
            return

        for root, _, files_in_dir in os.walk(cache_dir):
            for f_name in files_in_dir:
                if f_name.endswith('.pt'):
                    self.files.append(os.path.join(root, f_name))

        if self.verbose_dataset:
            print(f"CachedTensorDataset: Found {len(self.files)} .pt files in '{cache_dir}'.")
            if self.transform_pipeline and hasattr(self.transform_pipeline, 'stages'):
                print(
                    f"CachedTensorDataset: Using a transform pipeline with {len(self.transform_pipeline.stages)} stages.")
            elif self.transform_pipeline:
                print("CachedTensorDataset: Using a transform pipeline (structure unknown).")
            else:
                print("CachedTensorDataset: No transform pipeline provided.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Loads data for one index. Applies transformations via the pipeline.
        The pipeline is responsible for processing both X (tensor_stack) and
        Y (original_label_info) and returning the final X and the multi-task Y dictionary.

        Returns:
            Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
            - Processed X tensor.
            - Processed Y dictionary (e.g., {'regression_targets': ..., 'classification_targets': ...}).
            Returns None if file is missing, corrupted, doesn't contain expected keys,
            or if the pipeline fails or filters out the sample.
        """
        if idx >= len(self.files):
            if self.verbose_dataset:  # Should not happen with a well-behaved DataLoader
                print(f"CachedTensorDataset: Index {idx} out of bounds for {len(self.files)} files.")
            return None

        file_path = self.files[idx]
        try:
            # Consider weights_only=True for security and speed if files are trusted.
            data_dict = torch.load(file_path, map_location=torch.device('cpu'))

            tensor_stack = data_dict.get("tensor_stack")
            original_label_info = data_dict.get("label")  # This is the raw label dict from .pt file

            if tensor_stack is None or not isinstance(original_label_info, dict):
                if self.verbose_dataset:
                    print(
                        f"CachedTensorDataset: File {file_path} missing 'tensor_stack' or 'label' (as dict). Skipping.")
                return None
            if not isinstance(tensor_stack, torch.Tensor):
                if self.verbose_dataset:
                    print(
                        f"CachedTensorDataset: 'tensor_stack' in {file_path} is not a Tensor (got {type(tensor_stack)}). Skipping.")
                return None

            # X_initial is the tensor_stack, Y_initial is the raw label dictionary
            x_processed, y_processed_dict = tensor_stack, original_label_info

            if self.transform_pipeline:
                try:
                    # The pipeline's run method takes initial X and initial Y (raw label dict)
                    # and is expected to return processed X and processed Y (multi-task label dict)
                    x_processed, y_processed_dict = self.transform_pipeline.run(
                        tensor_stack,
                        original_label_info,
                        verbose_pipeline=False  # Stage-level verbosity is controlled by stage's init
                    )

                    # Validate that the pipeline (specifically LabelProcessorStage) produced the expected Y format
                    if not isinstance(y_processed_dict, dict) or \
                            'regression_targets' not in y_processed_dict or \
                            'classification_targets' not in y_processed_dict:
                        if self.verbose_dataset:
                            print(
                                f"CachedTensorDataset: Pipeline did not produce expected multi-task labels for {file_path}. "
                                f"Got Y type: {type(y_processed_dict)}, keys: {y_processed_dict.keys() if isinstance(y_processed_dict, dict) else 'N/A'}. Skipping.")
                        return None  # Skip sample if labels are not correctly processed

                except Exception as e_pipe:
                    if self.verbose_dataset:
                        print(f"CachedTensorDataset: Error in transform pipeline for {file_path}: {e_pipe}. Skipping.")
                    # import traceback # For debugging pipeline errors
                    # traceback.print_exc()
                    return None
            else:
                # If no pipeline, the original_label_info (raw dict) would be returned.
                # This is likely not what the multi-task model expects.
                # A LabelProcessorStage should typically always be part of the pipeline.
                if self.verbose_dataset:
                    print(f"CachedTensorDataset: No transform_pipeline provided for {file_path}. "
                          "Labels will be the raw dictionary from the .pt file. This might cause errors downstream.")
                # To be safe, if no pipeline, we can't guarantee y_processed_dict is in multi-task format.
                # For this setup, we assume a pipeline (with LabelProcessorStage) is essential.
                # If you want to support no pipeline, you'd need to handle raw labels differently.
                # Let's enforce that y_processed_dict must be in the correct format if no pipeline error.
                if not isinstance(y_processed_dict, dict) or \
                        'regression_targets' not in y_processed_dict or \
                        'classification_targets' not in y_processed_dict:
                    if self.verbose_dataset:
                        print(
                            f"CachedTensorDataset: Raw label for {file_path} is not in expected multi-task format and no pipeline was run. Skipping.")
                    return None

            return x_processed, y_processed_dict

        except FileNotFoundError:
            if self.verbose_dataset:
                print(f"CachedTensorDataset: File not found during __getitem__: {file_path}. Skipping.")
            return None
        except Exception as e:  # Catch other torch.load errors or unexpected issues
            if self.verbose_dataset:
                print(f"CachedTensorDataset: Error loading/processing file {file_path}: {e}. Skipping.")
            # import traceback
            # traceback.print_exc()
            return None
