import torch
import torch.nn as nn
import numpy as np
import random
from configs import BASE_CONFIG as config
import torch.nn.functional as F 

class HybridImpactModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 物理模型参数
        self.alpha = nn.Parameter(torch.tensor(1.5))  
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.lambda_ = nn.Parameter(torch.tensor(0.75))  # 初始值0.6
        self.sigma = torch.nn.Parameter(torch.tensor(1.0))  # 初始值设为1.0
        self.gamma =torch.nn.Parameter(torch.tensor(0.5))
        
        # 数据驱动模块
        self.feature_encoder = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
        self.target_feature_predictor = nn.Sequential(
            nn.Linear(5,32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64) # 输出: 和目标站点特征同维
        )
    
    def forward(self, target_geo, neighbor_geo, neighbor_history, wind_data, weather_data):
        # 物理权重计算
        dist = self.latlon_to_km(target_geo, neighbor_geo)
        theta = self.calc_wind_angle(target_geo, neighbor_geo, wind_data[1])
        # 高斯衰减项（sigma 是超参数，需预设或学习）
        sigma = self.sigma  
        gaussian_decay = torch.exp(-dist**2 / (2 * sigma**2))  # 形状 [n]
        dist1 = 1/dist
        # 风向修正项
        wind_adjustment = (1 + torch.cos(theta)) ** self.beta  # 形状 [n]
         # 逆幂律增强（1 + γ / dist^α）
        power_enhance = 1 + self.gamma / (dist**self.alpha + 1e-6)  # 避免除以0
        # 物理权重
        A_phys = gaussian_decay * wind_adjustment *power_enhance # 形状 [n]
       

        
        # 数据驱动权重计算
        _, (h_neighbor, _) = self.feature_encoder(neighbor_history)  # 周边站点特征
        h_neighbor = h_neighbor[-1]
        h_target = self.target_feature_predictor(
            # torch.cat([target_geo, weather_data], dim=-1)
            weather_data
        )  # 目标站点特征

 
        A_data = torch.cosine_similarity(h_target, h_neighbor, dim=-1)

        # 动态Top-k
        k = min(3, neighbor_geo.shape[0] // 2)
        A_phys = self.topk_normalize(A_phys,k)
        A_data = self.topk_normalize(A_data,k)

        # 混合权重
        A_hybrid = self.lambda_ * A_phys + (1 - self.lambda_) * A_data
        return A_hybrid, A_phys, A_data  # 修改：返回A_phys
    
    def latlon_to_km(self, target, neighbors):
        """
        计算目标点与多个邻居点的真实地理距离（单位：km）
        - target_geo: [lat, lon]（纬度, 经度）
        - neighbor_geo: [n, 2]（n个邻居的[lat, lon]）
        返回: [n] 的距离张量
        """
        # 将经纬度转换为弧度
        lat1, lon1 = torch.deg2rad(target[0]), torch.deg2rad(target[1])
        lat2 = torch.deg2rad(neighbors[:, 0])
        lon2 = torch.deg2rad(neighbors[:, 1])
        # 计算差值
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # Haversine 公式
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(a))
        dist = 6371.0 * c  # 地球半径 6371km
        return dist
    
    def calc_wind_angle(self, target, neighbor, wind_dir):
        # 计算站点连线与风向的夹角
        vec = neighbor - target
        angle = torch.atan2(vec[:,1], vec[:,0]) - wind_dir
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi  # 标准化到 [-π, π]
        return angle

    def topk_normalize(self, scores, k):

        topk_values, topk_indices = torch.topk(scores, k=k)

        topk_values_normalized = torch.softmax(topk_values,dim=-1)
        
        mask = torch.zeros_like(scores)
        mask[topk_indices] = topk_values_normalized
                  
        return mask