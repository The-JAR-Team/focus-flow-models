import os
from torch.utils.data import Dataset
import torch


class CachedTensorDataset(Dataset):
    def __init__(self, cache_dir):
        """
        Recursively collects all .pt files in cache_dir.
        Assumes each file is a dict with keys "tensor_stack" and "label".
        """
        self.cache_dir = cache_dir
        self.files = []
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                if f.endswith('.pt'):
                    self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data["tensor_stack"], data["label"]
