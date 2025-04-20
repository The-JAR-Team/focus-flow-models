import os
import torch
from torch.utils.data import Dataset
import pandas as pd # Keep pandas import if used elsewhere or for future type hints


class CachedTensorDataset(Dataset):
    def __init__(self, cache_dir):
        """
        Recursively collects all .pt files in cache_dir.
        Assumes each file is a dict with keys "tensor_stack" and "label".
        """
        self.cache_dir = cache_dir
        self.files = []
        if not os.path.isdir(cache_dir):
            return # Initialize with empty list

        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                if f.endswith('.pt'):
                    self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads data for one index. Returns None if file is missing,
        corrupted, or doesn't contain expected keys.
        """
        file_path = self.files[idx]
        try:
            # Use map_location='cpu' if files were saved on GPU but loading on CPU
            data = torch.load(file_path, map_location=torch.device('cpu'))

            # Use .get() for safer access, check if either is None
            tensor = data.get("tensor_stack")
            label = data.get("label")

            if tensor is None or label is None:
                 # Don't print warning for every bad file, gets too noisy.
                 # Logging could be used here instead.
                 # print(f"Warning: File {file_path} missing 'tensor_stack' or 'label' key. Skipping.")
                 return None # Return None for bad files

            return tensor, label

        except FileNotFoundError:
            # print(f"Warning: File not found during __getitem__: {file_path}. Skipping.")
            return None # Return None for missing files
        except Exception as e: # Catch other load errors (like UnpicklingError, RuntimeError)
            # print(f"Warning: Error loading/processing file {file_path}: {e}. Skipping.")
            return None # Return None for corrupted/unreadable files

