import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import string
import re

class FruitDataset(Dataset):
    def __init__(self, data, seq_len=25):
        self.seq_len = seq_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)