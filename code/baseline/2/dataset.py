# import pickle
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import os
# import math
# import numpy as np
# from math import atan2, degrees, radians, cos, exp

# class AirQualityDataset(Dataset):
#     def __init__(self, pkl_file, distances, config, mode="hybrid"):
#         self.config = config  # 存储当前实验配置
#         with open(pkl_file, "rb") as f:
#             self.samples = pickle.load(f)
#         self.distances = distances
#         self.mode = mode

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         # 图像序列
#         imgs = []
#         for img_path in sample['images']:
#             if os.path.exists(img_path):
#                 img = Image.open(img_path).convert('RGB')
#                 img = self.transform(img)
#                 imgs.append(img)
#             else:
#                 imgs.append(torch.zeros(3, 224, 224))
#         imgs = torch.stack(imgs) if imgs else torch.zeros(1, 3, 224, 224)

#         # 污染物序列
#         sample['pollution_seq'] = sample['pollution_seq'][:self.config["site_nums"],-self.config["history_hours"]:,:]
#         pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

#         # 气象序列
#         sample['weather_seq'] = sample['weather_seq'][-self.config["history_hours"]:,:]
#         weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

#         # 目标值
#         target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

#         # 动态邻接矩阵
#         adj_mask = torch.ones(len(self.distances))
#         if self.config["dynamic_edge"]:
#             adj_mask = self.build_dynamic_edges(sample, adj_mask)
        
#         return imgs, pollution_seq, weather_seq, adj_mask, target


#     #高斯衰减，距离，风向  动态
#     def build_dynamic_edges(self, sample, adj_mask):
#         from math import atan2, degrees, radians
#         import numpy as np

#         coords = self.config["station_coords"]
#         target_lon, target_lat = coords["bjdst"]

#         sigma = 5  # 高斯距离参数
#         edge_weights = []

#         wind_dir = sample["weather_seq"][-1][1] if self.config["dynamic_use_wind"] else None
#         wind_speed = sample["weather_seq"][-1][2] if self.config["dynamic_use_wind"] else None

#         for name, (lon, lat) in coords.items():
#             if name == "bjdst":
#                 continue
#             # 1. 计算距离  高斯衰减
#             dx = target_lon - lon
#             dy = target_lat - lat
#             dist = np.sqrt(dx**2 + dy**2) * 111  # 粗略换算为 km
#             dist_weight = np.exp(-dist**2 / (2 * sigma**2))
#             # print(f"Distance from dfmz to {name}: {dist:.2f} km, weight: {dist_weight:.4f}")

#             # 2. 计算风向夹角（单位：度）
#             # 版本1  0 或  1
#             # if wind_dir is not None:
#             #     angle = degrees(atan2(dy, dx)) % 360
#             #     delta = abs((angle - wind_dir + 180) % 360 - 180)
#             #     wind_mask = 1.0 if delta < 90 else 0.0
#             # else:
#             #     wind_mask = 1.0

#             # 版本2  0-1 余弦权重
#             if wind_dir != 0.0:
#                 angle = degrees(atan2(dy, dx)) % 360
#                 delta = abs((angle - wind_dir + 180) % 360 - 180)  # 最小角度差 [0°, 180°]
#                 wind_mask = cos(radians(delta)) if delta < 90 else 0.0  # 使用余弦函数，delta=0°时权重=1，delta=90°时权重=0
#             else:
#                 wind_mask = 1  # 无风向时，默认权重为1
            
#             #版本3  0-1 高斯衰减权重
#             # if wind_dir is not None:
#             #     angle = degrees(atan2(dy, dx)) % 360
#             #     delta = abs((angle - wind_dir + 180) % 360 - 180)  # 最小角度差 [0°, 180°]
#             #     sigma = 45  # 控制衰减速度，sigma越小，衰减越快
#             #     wind_mask = exp(-(delta ** 2) / (2 * sigma ** 2))  # 高斯衰减，delta=0°时权重=1
#             # else:
#             #     wind_mask = 1.0  # 无风向时，默认权重为1

#             # print(f"Wind direction to {name}: {wind_dir}, angle: {angle:.2f}, wind mask: {wind_mask}")
#             # vmax=7.0
#             # wind_speed_norm = min(wind_speed / vmax, 1.0)
#             # wind_speed_norm = np.log1p(wind_speed) / np.log1p(10.0)  # 归一化到 [0, 1]


#             # wind_speed_norm = np.tanh(wind_speed)

#             if wind_speed != 0:
#                 wind_speed_norm = wind_speed / (1 + wind_speed)
#             else:
#                 wind_speed_norm = 1
#             edge_weights.append(dist_weight * wind_mask * wind_speed_norm)
#             # print(f"Edge weight to {name}: {edge_weights[-1]:.4f}")

#         return torch.tensor(edge_weights, dtype=torch.float32)












import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from math import atan2, degrees, radians, cos, exp

class AirQualityDataset(Dataset):
    def __init__(self, pkl_file, config, mode="hybrid"):
        self.config = config
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 站点信息
        self.coords = config["station_coords"]
        self.target_name = config.get("target_name", "bjdst")
        self.target_idx = list(self.coords.keys()).index(self.target_name)
        self.site_nums = config["site_nums"]

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

        # 污染物序列 [N, T, F]
        pol_seq = sample['pollution_seq'][:self.site_nums, -self.config["history_hours"]:, :]
        pol_seq = torch.tensor(pol_seq, dtype=torch.float32)

        # 气象序列 [T, F_w]
        wea_seq = sample['weather_seq'][-self.config["history_hours"]:, :]
        wea_seq = torch.tensor(wea_seq, dtype=torch.float32)

        # 目标值
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        # 邻接矩阵 [N, N]
        A = torch.eye(self.site_nums, dtype=torch.float32)
        if self.config.get("dynamic_edge", True):
            A = self.build_dynamic_adj(sample)

        return imgs, pol_seq, wea_seq, A, target

    def build_dynamic_adj(self, sample):
        N = self.site_nums
        A = np.zeros((N, N), dtype=np.float32)

        coords = self.coords
        target_name = self.target_name
        target_lon, target_lat = coords[target_name]

        sigma = 5  # 高斯距离参数

        wind_dir = sample["weather_seq"][-1][1] if self.config.get("dynamic_use_wind", True) else None
        wind_speed = sample["weather_seq"][-1][2] if self.config.get("dynamic_use_wind", True) else None

        for i, (name, (lon, lat)) in enumerate(coords.items()):
            if name == target_name:
                continue
            # 距离权重
            dx, dy = target_lon - lon, target_lat - lat
            dist = np.sqrt(dx**2 + dy**2) * 111  # 转 km
            dist_weight = np.exp(-dist**2 / (2 * sigma**2))

            # 风向权重 (cos 修正)
            if wind_dir is not None and wind_dir != 0.0:
                angle = degrees(atan2(dy, dx)) % 360
                delta = abs((angle - wind_dir + 180) % 360 - 180)
                wind_mask = cos(radians(delta)) if delta < 90 else 0.0
            else:
                wind_mask = 1.0

            # 风速归一化
            wind_speed_norm = wind_speed / (1 + wind_speed) if wind_speed and wind_speed > 0 else 1.0

            weight = dist_weight * wind_mask * wind_speed_norm

            # 对称邻接矩阵
            target_idx = list(coords.keys()).index(target_name)
            A[target_idx, i] = weight
            A[i, target_idx] = weight

        # 加自环
        for i in range(N):
            A[i, i] = 1.0

        return torch.tensor(A, dtype=torch.float32)
