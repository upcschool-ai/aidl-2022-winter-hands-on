import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, n_samples, n_features, n_outputs):
        super().__init__()
        self.data = torch.randn(n_samples, n_features)
        self.labels = torch.zeros(n_samples, n_outputs)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
