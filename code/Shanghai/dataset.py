import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np
from math import atan2, degrees, radians, cos, exp
from HybridImpactModel import HybridImpactModel

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

        # 图像序列
        imgs = []
        for img_path in sample['images']:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(torch.zeros(3, 224, 224))
        imgs = torch.stack(imgs) if imgs else torch.zeros(1, 3, 224, 224)

        # 污染物序列
        sample['pollution_seq'] = sample['pollution_seq'][:self.config["site_nums"],-self.config["history_hours"]:,:]
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

        # 气象序列（保留，因为PollutionGATEncoder需要）
        sample['weather_seq'] = sample['weather_seq'][-self.config["history_hours"]:,:]
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

        # 目标值
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        # 动态邻接矩阵
        adj_mask, adj_phys = torch.ones(len(self.distances)), torch.ones(len(self.distances))
        if self.config["dynamic_edge"]:
            adj_mask, adj_phys = self.build_weights(sample)
        
        # Squeeze if needed (from [1,12] to [12])
        if adj_mask.dim() > 1:
            adj_mask = adj_mask.squeeze(0)
        if adj_phys.dim() > 1:
            adj_phys = adj_phys.squeeze(0)

        return imgs, pollution_seq, weather_seq, (adj_mask, adj_phys), target

    def build_dynamic_edges(self, sample, adj_mask):
        from math import atan2, degrees, radians
        import numpy as np

        coords = self.config["station_coords"]
        target_lon, target_lat = coords["dfmz"]

        sigma = 5
        edge_weights = []

        wind_dir = sample["weather_seq"][-1][1] if self.config["dynamic_use_wind"] else None
        wind_speed = sample["weather_seq"][-1][2] if self.config["dynamic_use_wind"] else None
        for name, (lon, lat) in coords.items():
            if name == "dfmz":
                continue
            dx = target_lon - lon
            dy = target_lat - lat
            dist = np.sqrt(dx**2 + dy**2) * 111
            dist_weight = np.round(np.exp(-dist**2 / (2 * sigma**2)), 3)
        
            if wind_dir is not None:
                angle = degrees(atan2(dy, dx)) % 360
                delta = abs((angle - wind_dir + 180) % 360 - 180)
                wind_mask = 1.0 if delta < 90 else 0.0
            else:
                wind_mask = 1.0
          
            edge_weights.append(dist_weight * wind_mask)

        return torch.tensor(edge_weights, dtype=torch.float32)
    
    def build_weights(self, sample):
        edge_weights = []
        target_geo = torch.tensor([31.2397, 121.4998])  # 经纬度
        neighbor_geo = torch.tensor([ 
                                        [31.111, 121.567],   # Shiwuchang
                                        [31.2715, 121.4800],  # Hongkou
                                        [31.1613, 121.4208],  # Shangshida
                                        [31.2728, 121.5306],  # Yangpu
                                        [31.1514, 121.1139],  # Qingpu
                                        [31.2230, 121.4456],  # Jingan
                                        [31.1869, 121.6986],  # PDchuansha
                                        [31.2105, 121.5508],  # PDxinqu
                                        [31.2012, 121.5874]   # PDzhangjiang
                                    ])  # 周边站点经纬度
        neighbor_history = torch.from_numpy(sample["pollution_seq"]).float()
        wind_dir = sample["weather_seq"][-1][1]
        wind_speed = sample["weather_seq"][-1][2]
        weather_data = sample["weather_seq"][-1]
        wind_data = torch.tensor([wind_speed.item(), wind_dir.item()])
        weather_data = torch.tensor(weather_data)
        model = HybridImpactModel(input_dim=6)
        impact_scores, phy_scores, data_scores = model(target_geo, neighbor_geo, neighbor_history, wind_data, weather_data)
        return impact_scores, phy_scores