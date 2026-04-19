from pathlib import Path; import sys
COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path: sys.path.insert(0, str(COMMON_DIR))

import torch
import torch.nn as nn
from SCSA import SCSA
from PollutionEncoder import PollutionGATEncoder
from HybridImpactModel import HybridImpactModel


class AirQualityModel(nn.Module):
    def __init__(self, config, num_pollution_feat=6, num_weather_feat=5):
        super().__init__()
        self.config = config
        self.use_image = self.config.get("use_image", True)
        self.use_pollution = self.config.get("use_pollution", True)

        img_dim = self.config["img_hidden_dim"]
        pollution_dim = self.config["pollution_hidden_dim"]

        # image branch (V-RSCSA)
        if self.use_image:
            if self.config["cnn_backbone"] == "resnet18":
                from torchvision.models import resnet18, ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if self.config.get("use_pretrained", True) else None
                base = resnet18(weights=weights)
                self.cnn = nn.Sequential(
                    nn.Sequential(
                        *list(base.children())[:6],
                        SCSA(dim=128, head_num=4),
                        *list(base.children())[6:-2],
                        SCSA(dim=512, head_num=8)
                    ),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                cnn_out_dim = 512
            else:
                raise ValueError(f"Unsupported cnn_backbone: {self.config['cnn_backbone']}")

            self.img_rnn = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=img_dim,
                batch_first=True
            )

        # spatiotemporal branch (P-STEM)
        if self.use_pollution:
            self.impact_model = HybridImpactModel()
            self.pollution_encoder = PollutionGATEncoder(
                in_pollution=num_pollution_feat,
                hidden_dim=pollution_dim,
                weather_dim=num_weather_feat
            )

        # gated fusion
        if self.use_image and self.use_pollution:
            self.gate_network = nn.Sequential(
                nn.Linear(img_dim + pollution_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            final_fusion_dim = img_dim + pollution_dim
        else:
            final_fusion_dim = 0
            if self.use_image: final_fusion_dim += img_dim
            if self.use_pollution: final_fusion_dim += pollution_dim

        # regressor head
        self.regressor = nn.Sequential(
            nn.Linear(final_fusion_dim, self.config["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config["mlp_hidden_dim"], 1)
        )

    def get_coords(self, B):
        station_coords_dict = self.config["station_coords"]
        target_key = self.config["target_station_key"]
        station_list = [station_coords_dict[target_key]] + \
            [v for k, v in sorted(station_coords_dict.items()) if k != target_key]
        coords = torch.tensor(station_list, dtype=torch.float32)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def forward(self, imgs, pollution, weather, wind_info=None, return_contribution=False):
        img_feat, num_feat = None, None
        B = imgs.size(0) if self.use_image else pollution.size(0)

        # spatiotemporal branch (P-STEM)
        if self.use_pollution:
            coords = self.get_coords(B).to(pollution.device)
            station_coords_dict = self.config["station_coords"]
            target_key = self.config["target_station_key"]
            target_coord = torch.tensor(station_coords_dict[target_key], dtype=torch.float32, device=pollution.device)
            neighbor_coords = {k: v for k, v in sorted(station_coords_dict.items()) if k != target_key}
            neighbor_coord = torch.tensor(
                [v for v in neighbor_coords.values()],
                dtype=torch.float32, device=pollution.device
            )

            if self.config["dynamic_edge"] and wind_info is not None:
                wind_speed, wind_dir = wind_info
                wind_dir_rad = wind_dir * (torch.pi / 180.0)
                wind_data = torch.stack([wind_speed, wind_dir_rad], dim=1)

                target_geo = target_coord.unsqueeze(0).expand(B, -1)
                neighbor_geo = neighbor_coord.unsqueeze(0).expand(B, -1, -1)

                adj_phys = self.impact_model(
                    target_geo, neighbor_geo, wind_data
                )
            else:
                adj_phys = torch.ones(B, self.config["site_nums"], device=pollution.device)

            num_feat = self.pollution_encoder(pollution, weather, coords, adj_phys)

        # image branch (V-RSCSA)
        if self.use_image:
            B, T, C, H, W = imgs.shape
            imgs_reshaped = imgs.view(B * T, C, H, W)
            cnn_feats = self.cnn(imgs_reshaped)
            cnn_feats = cnn_feats.view(B, T, -1)
            _, (h_img, _) = self.img_rnn(cnn_feats)
            img_feat = h_img[-1]

        # fusion
        alpha = None
        if self.use_image and self.use_pollution:
            gate_input = torch.cat([img_feat, num_feat], dim=1)
            alpha = self.gate_network(gate_input)
            f_all = torch.cat([alpha * img_feat, (1 - alpha) * num_feat], dim=1)
        else:
            feats = []
            if img_feat is not None: feats.append(img_feat)
            if num_feat is not None: feats.append(num_feat)
            if not feats: raise ValueError("At least one branch must be enabled")
            f_all = torch.cat(feats, dim=1)
            if self.use_image and not self.use_pollution:
                alpha = torch.ones(B, 1, device=f_all.device)
            elif not self.use_image and self.use_pollution:
                alpha = torch.zeros(B, 1, device=f_all.device)

        # regression
        output = self.regressor(f_all)

        if self.training and self.config.get("use_contrastive", False):
            if img_feat is None or num_feat is None:
                raise ValueError("Contrastive learning requires both image and pollution branches to be enabled.")
            return output, img_feat, num_feat

        if return_contribution:
            if alpha is None:
                alpha = torch.full((B, 1), -1.0, device=f_all.device)
            return output, alpha

        return output