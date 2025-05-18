import os
from typing import Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import pandas as pd # Keep pandas import if used elsewhere or for future type hints

from Preprocess.Pipeline import OrchestrationPipeline


class CachedTensorDataset(Dataset):
    def __init__(self,
                 cache_dir: str,
                 transform_pipeline: Optional[OrchestrationPipeline] = None,
                 verbose_dataset: bool = False):
        """
        Recursively collects all .pt files in cache_dir.
        Assumes each file is a dict with keys "tensor_stack" and "label".
        Applies an optional transformation pipeline to the tensor_stack.

        Args:
            cache_dir (str): Directory containing the cached .pt files.
            transform_pipeline (Optional[OrchestrationPipeline]): An OrchestrationPipeline
                instance to process the 'tensor_stack' after loading. If None,
                no transformations are applied.
            verbose_dataset (bool): If True, dataset-level messages are printed.
                                   Individual pipeline stages might have their own verbosity.
        """
        self.cache_dir = cache_dir
        self.files = []
        self.transform_pipeline = transform_pipeline
        self.verbose_dataset = verbose_dataset

        if not os.path.isdir(cache_dir):
            if self.verbose_dataset:
                print(f"CachedTensorDataset: Cache directory '{cache_dir}' not found. Dataset will be empty.")
            return

        for root, dirs, files_in_dir in os.walk(cache_dir):  # Renamed 'files' to 'files_in_dir'
            for f in files_in_dir:
                if f.endswith('.pt'):
                    self.files.append(os.path.join(root, f))

        if self.verbose_dataset:
            print(f"CachedTensorDataset: Found {len(self.files)} .pt files in '{cache_dir}'.")
            if self.transform_pipeline:
                print(
                    f"CachedTensorDataset: Using a transform pipeline with {len(self.transform_pipeline.stages)} stages.")
            else:
                print("CachedTensorDataset: No transform pipeline provided.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, Any]]:
        """
        Loads data for one index. Applies transformations if a pipeline is provided.
        Returns None if file is missing, corrupted, doesn't contain expected keys,
        or if a transformation fails expectedly (though stages should ideally handle errors).
        """
        if idx >= len(self.files):
            # This case should ideally not be reached if DataLoader handles indices correctly
            if self.verbose_dataset:
                print(f"CachedTensorDataset: Index {idx} out of bounds for {len(self.files)} files.")
            return None

        file_path = self.files[idx]
        try:
            data_dict = torch.load(file_path, map_location=torch.device('cpu'))

            tensor_stack = data_dict.get("tensor_stack")
            label = data_dict.get("label")

            if tensor_stack is None or label is None:
                if self.verbose_dataset:  # Only print if dataset verbosity is on
                    print(f"CachedTensorDataset: File {file_path} missing 'tensor_stack' or 'label' key. Skipping.")
                return None

            # Apply transformations if a pipeline is provided
            if self.transform_pipeline:
                # The pipeline's run method should handle its own verbosity based on its stages
                # We pass verbose=False here to let stages control their own printouts,
                # or you can pass self.verbose_dataset to make pipeline stages verbose too.
                tensor_stack = self.transform_pipeline.run(tensor_stack,
                                                           verbose=False)  # Or verbose=self.verbose_dataset

            return tensor_stack, label

        except FileNotFoundError:
            if self.verbose_dataset:
                print(f"CachedTensorDataset: File not found during __getitem__: {file_path}. Skipping.")
            return None
        except Exception as e:
            if self.verbose_dataset:
                print(f"CachedTensorDataset: Error loading/processing file {file_path}: {e}. Skipping.")
            return None

