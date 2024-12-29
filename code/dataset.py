import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class LabeledDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        graph_id = self.data_list[idx][0]
        expr = self.data_list[idx][1]
        label = self.data_list[idx][2]
        return graph_id, expr, label

class UnlabeledDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        expr = self.data_list[idx]
        return expr
    