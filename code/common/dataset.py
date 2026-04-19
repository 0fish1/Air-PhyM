import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class AirQualityDataset(Dataset):
    def __init__(self, pkl_file, distances, config, mode="hybrid"):
        self.config = config
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        self.distances = distances
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # image sequence
        imgs = []
        for img_path in sample['images']:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(torch.zeros(3, 224, 224))
        imgs = torch.stack(imgs) if imgs else torch.zeros(1, 3, 224, 224)

        # pollution sequence
        sample['pollution_seq'] = sample['pollution_seq'][:self.config["site_nums"], -self.config["history_hours"]:,:]
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

        # weather sequence
        sample['weather_seq'] = sample['weather_seq'][-self.config["history_hours"]:,:]
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

        # target
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        # wind info for dynamic adjacency
        wind_info = None
        if self.config["dynamic_edge"]:
            wind_dir = sample["weather_seq"][-1][1]
            wind_speed = sample["weather_seq"][-1][2]
            wind_info = (torch.tensor(wind_speed, dtype=torch.float32),
                         torch.tensor(wind_dir, dtype=torch.float32))

        return imgs, pollution_seq, weather_seq, wind_info, target