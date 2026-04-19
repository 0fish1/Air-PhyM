import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridImpactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.tau = nn.Parameter(torch.tensor(5.0))

    def forward(self, target_geo, neighbor_geo, wind_data):
        """
        Args:
            target_geo: [B, 2] (lon, lat)
            neighbor_geo: [B, N, 2] (lon, lat)
            wind_data: [B, 2] (wind_speed, wind_dir_rad)
        Returns:
            A_phys: [B, N]
        """
        dist = self.latlon_to_km(target_geo, neighbor_geo)  # [B, N]
        theta = self.calc_wind_angle(target_geo, neighbor_geo, wind_data[:, 1])  # [B, N]
        wind_speed = wind_data[:, 0]  # [B]

        gaussian_decay = torch.exp(-dist**2 / (2 * self.tau**2))  # [B, N]

        advection = wind_speed.unsqueeze(1) / (dist + 1e-6) * torch.cos(theta)  # [B, N]
        advection_enhance = 1 + self.beta * torch.relu(advection)  # [B, N]

        A_phys = gaussian_decay * advection_enhance  # [B, N]
        return A_phys

    def latlon_to_km(self, target, neighbors):
        """
        target: [B, 2] (lon, lat)
        neighbors: [B, N, 2] (lon, lat)
        Returns: [B, N]
        """
        lat1, lon1 = torch.deg2rad(target[:, 1]), torch.deg2rad(target[:, 0])
        lat2 = torch.deg2rad(neighbors[:, :, 1])
        lon2 = torch.deg2rad(neighbors[:, :, 0])
        dlat = lat2 - lat1.unsqueeze(1)
        dlon = lon2 - lon1.unsqueeze(1)
        a = torch.sin(dlat/2)**2 + torch.cos(lat1.unsqueeze(1)) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(a + 1e-12))
        dist = 6371.0 * c
        return dist

    def calc_wind_angle(self, target, neighbor, wind_dir):
        """
        target: [B, 2] (lon, lat)
        neighbor: [B, N, 2] (lon, lat)
        wind_dir: [B] radian wind direction
        Returns: [B, N]
        """
        vec = neighbor - target.unsqueeze(1)  # [B, N, 2]
        angle = torch.atan2(vec[:, :, 1], vec[:, :, 0]) - wind_dir.unsqueeze(1)  # [B, N]
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        return angle