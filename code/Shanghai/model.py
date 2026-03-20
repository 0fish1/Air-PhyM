import torch
import torch.nn as nn
from torchvision import models
from SCSA import SCSA
from PollutionEncoder import PollutionGATEncoder

# ：交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, output_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 线性变换：Q（图像嵌入），K、V（数值嵌入）
        self.q_linear = nn.Linear(query_dim, output_dim)
        self.k_linear = nn.Linear(key_value_dim, output_dim)
        self.v_linear = nn.Linear(key_value_dim, output_dim)
        
        self.out_linear = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        """
        query: 图像嵌入 [B, query_dim]
        key, value: 数值嵌入 [B, key_value_dim]
        """
        B = query.size(0)
        
        # 线性变换生成 Q, K, V
        Q = self.q_linear(query)  # [B, output_dim]
        K = self.k_linear(key)    # [B, output_dim]
        V = self.v_linear(value)  # [B, output_dim]
        
        # 调整形状以支持多头注意力
        Q = Q.view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]
        K = K.view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]
        V = V.view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]
        
        # 计算注意力分数
        scores = torch.einsum('bhd,bkd->bhk', Q, K) * self.scale  # [B, heads, 1]
        attn = torch.softmax(scores, dim=-1)  # [B, heads, 1]
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.einsum('bhk,bhd->bhd', attn, V)  # [B, heads, head_dim]
        out = out.reshape(B, -1)  # [B, output_dim]
        
        # 输出投影
        out = self.out_linear(out)  # [B, output_dim]
        return out

class AirQualityModel(nn.Module):
    def __init__(self, config, num_pollution_feat=6, num_weather_feat=5, hidden_dim=128):
        super().__init__()
        self.config = config
        self.use_image = self.config["use_image"]
        self.use_pollution = self.config["use_pollution"]

        # 图像分支
        if self.use_image:
            if self.config["cnn_backbone"] == "resnet18":
                from torchvision.models import resnet18, ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if self.config.get("use_pretrained", True) else None
                base = resnet18(weights=weights)
                base.fc = nn.Identity()
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
                raise ValueError("Invalid cnn_backbone")

            self.img_rnn = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=self.config["img_hidden_dim"],
                batch_first=True
            )

        # 污染物分支
        if self.use_pollution:
            self.pollution_encoder = PollutionGATEncoder(
                in_pollution=num_pollution_feat,
                hidden_dim=hidden_dim,
                tcn_channels=[8,16,32,64],
                weather_dim=num_weather_feat,
                pos_dim=8,
                gat_heads=4
            )

        # 融合 & 回归
        total_dim = 0
        if self.use_image:
            total_dim += self.config["img_hidden_dim"]
        if self.use_pollution:
            total_dim += self.config["pollution_hidden_dim"]

        # 修改：替换 nn.Identity() 为交叉注意力
        if self.config["fusion_type"] == "cross_attention":
            self.fusion = CrossAttention(
                query_dim=self.config["img_hidden_dim"],
                key_value_dim=self.config["pollution_hidden_dim"],
                output_dim=total_dim,  # 保持与 concat 相同的输出维度
                num_heads=1
            )
        elif self.config["fusion_type"] == "concat":
            self.fusion = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(total_dim, self.config["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["mlp_hidden_dim"], 1)
        )

    def get_coords(self, B):
        station_list = [                (31.2397, 121.4998),
                                        (31.111, 121.567),   # Shiwuchang
                                        (31.2715, 121.4800),  # Hongkou
                                        (31.1613, 121.4208),  # Shangshida
                                        (31.2728, 121.5306),  # Yangpu
                                        (31.1514, 121.1139),  # Qingpu
                                        (31.2230, 121.4456),  # Jingan
                                        (31.1869, 121.6986),  # PDchuansha
                                        (31.2105, 121.5508),  # PDxinqu
                                        (31.2012, 121.5874)   # PDzhangjiang
                                    ]
        coords = torch.tensor(station_list, dtype=torch.float32)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def forward(self, imgs, pollution, weather, adj_mask=None):
        feats = []
        img_feat, num_feat = None, None

        B = imgs.size(0) if self.use_image else pollution.size(0)

        # 污染物分支
        if self.use_pollution:
            adj_hybrid, adj_phys = adj_mask
            coords = self.get_coords(B).to(pollution.device)
            f_pollution = self.pollution_encoder(pollution, weather, coords, adj_phys,
                                                save_dir=self.config["save_dir"], batch_idx=0)
            feats.append(f_pollution)
            num_feat = f_pollution

        # 图像分支
        if self.use_image:
            B, T, C, H, W = imgs.shape
            imgs = imgs.view(B * T, C, H, W)
            cnn_feats = self.cnn(imgs)
            cnn_feats = cnn_feats.view(B, T, -1)
            _, (h_img, _) = self.img_rnn(cnn_feats)
            feats.append(h_img[-1])
            img_feat = h_img[-1]

        # 修改：使用交叉注意力进行融合
        if self.config["fusion_type"] == "cross_attention":
            if self.use_image and self.use_pollution:
                f_all = self.fusion(img_feat, num_feat, num_feat)  # Q=img, K=V=num
            elif self.use_image:
                f_all = img_feat  # 仅图像分支
            elif self.use_pollution:
                f_all = num_feat  # 仅污染物分支
            else:
                raise ValueError("At least one branch (image or pollution) must be enabled")
        else:
            f_all = torch.cat(feats, dim=1)  # 原有的 concat 融合

        output = self.regressor(f_all)

        if self.training and self.config.get("use_contrastive", False):
            return output, img_feat, num_feat

        return output