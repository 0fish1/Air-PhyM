import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np
from math import atan2, degrees, radians, cos, exp

class AirQualityDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        neighbor_pm25 = sample['pollution_seq'][:, -1:, :][:, 0, 0].tolist()
        # pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

        # 目标值
        target_pm25 = sample['target']

        return neighbor_pm25, target_pm25





