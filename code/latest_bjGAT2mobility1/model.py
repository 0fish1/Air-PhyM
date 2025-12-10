import torch
import torch.nn as nn
from torchvision import models
from SCSA import SCSA  
from PollutionEncoder import PollutionGATEncoder # 请确保文件名正确

# =============================================================================
# 1. 交叉注意力模块 (Cross-Attention Module)
# =============================================================================
class CrossAttention(nn.Module):
    """
    一个标准的多头交叉注意力模块。
    它接收一个 query 序列和一个 key/value 序列，输出一个被 query 加权过的 value 序列。
    """
    def __init__(self, query_dim, key_value_dim, embed_dim, num_heads=4):
        super().__init__()
        # 使用PyTorch内置的高效多头注意力实现
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # 线性投影层，将输入维度映射到统一的注意力嵌入维度 (embed_dim)
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_value_dim, embed_dim)
        self.v_proj = nn.Linear(key_value_dim, embed_dim)

    def forward(self, query, key, value):
        """
        Args:
            query (torch.Tensor): 查询张量, shape [B, query_dim]
            key (torch.Tensor): 键张量, shape [B, key_value_dim]
            value (torch.Tensor): 值张量, shape [B, key_value_dim]
        
        Returns:
            torch.Tensor: 注意力加权后的输出, shape [B, embed_dim]
        """
        # 为输入增加一个序列长度维度 (L=1)，以符合 nn.MultiheadAttention 的输入要求 [B, L, D]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        
        # 将输入投影到注意力空间
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 计算多头注意力
        # attn_output 的维度是 [B, 1, embed_dim]
        attn_output, _ = self.multihead_attn(q, k, v)
        
        # 移除序列长度维度，返回 [B, embed_dim]
        return attn_output.squeeze(1)

# =============================================================================
# 2. 主模型 (AirQualityModel)
# =============================================================================
class AirQualityModel(nn.Module):
    def __init__(self, config, num_pollution_feat=6, num_weather_feat=5):
        super().__init__()
        self.config = config
        self.use_image = self.config.get("use_image", True)
        self.use_pollution = self.config.get("use_pollution", True)
        
        img_dim = self.config["img_hidden_dim"]
        pollution_dim = self.config["pollution_hidden_dim"] 

        # -------------------------------
        # 图像分支 (V-RSCSA)
        # -------------------------------
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
                    
        # -------------------------------
        # 时空分支 (P-STEM)
        # -------------------------------
        if self.use_pollution:
            self.pollution_encoder = PollutionGATEncoder(
                in_pollution=num_pollution_feat,
                hidden_dim=pollution_dim,
                weather_dim=num_weather_feat
                # 确保 PollutionGATEncoder 的参数与此匹配
            )

        # -------------------------------
        # 融合模块 (Fusion Module)
        # -------------------------------
        self.fusion_type = self.config.get("fusion_type", "concat") 
        
        if self.fusion_type == "symmetrical_gated_attention" and self.use_image and self.use_pollution:
            # 定义一个统一的融合/注意力空间维度
            fusion_dim = 128 
            
            # 1. 两个方向的交叉注意力实例
            # (img -> num): 用图像特征增强时空特征
            self.cross_attn_i2n = CrossAttention(
                query_dim=img_dim, key_value_dim=pollution_dim, embed_dim=fusion_dim, num_heads=4
            )
            # (num -> img): 用时空特征增强图像特征
            self.cross_attn_n2i = CrossAttention(
                query_dim=pollution_dim, key_value_dim=img_dim, embed_dim=fusion_dim, num_heads=4
            )
            
            # 2. 门控网络，用于生成贡献度权重 alpha
            self.gate_network = nn.Sequential(
                nn.Linear(img_dim + pollution_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            final_fusion_dim = fusion_dim
        else: # 简单拼接 (concat) 或单模态情况
            final_fusion_dim = 0
            if self.use_image: final_fusion_dim += img_dim
            if self.use_pollution: final_fusion_dim += pollution_dim
        
        # -------------------------------
        # 回归头 (Regressor Head)
        # -------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(final_fusion_dim, self.config["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.2), # 可以在回归头中加入Dropout防止过拟合
            nn.Linear(self.config["mlp_hidden_dim"], 1)
        )

    def get_coords(self, B):
        # 从config中获取站点坐标
        station_coords_dict = self.config["station_coords"]
        # 确保 'bjdst' 是第一个
        station_list = [station_coords_dict['bjdst']] + [v for k, v in sorted(station_coords_dict.items()) if k != 'bjdst']
        coords = torch.tensor(station_list, dtype=torch.float32)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def forward(self, imgs, pollution, weather, adj_mask=None, return_contribution=False):
        img_feat, num_feat = None, None
        B = imgs.size(0) if self.use_image else pollution.size(0)

        # --- 1. 特征提取 ---
        # 时空分支 (P-STEM)
        if self.use_pollution:
            _, adj_phys = adj_mask
            coords = self.get_coords(B).to(pollution.device)
            num_feat = self.pollution_encoder(pollution, weather, coords, adj_phys)

        # 图像分支 (V-RSCSA)
        if self.use_image:
            B, T, C, H, W = imgs.shape
            imgs_reshaped = imgs.view(B * T, C, H, W)
            cnn_feats = self.cnn(imgs_reshaped)
            cnn_feats = cnn_feats.view(B, T, -1)
            _, (h_img, _) = self.img_rnn(cnn_feats)
            img_feat = h_img[-1]

        # --- 2. 特征融合 ---
        alpha = None
        # 对称门控交叉注意力融合
        if self.fusion_type == "symmetrical_gated_attention" and self.use_image and self.use_pollution:
            # a. 双向增强
            attended_num_feat = self.cross_attn_i2n(query=img_feat, key=num_feat, value=num_feat)
            attended_img_feat = self.cross_attn_n2i(query=num_feat, key=img_feat, value=img_feat)
            
            # b. 计算门控权重 alpha (attended_img_feat 的权重)
            # 使用detach确保门控网络的学习不直接影响主干分支的梯度
            gate_input = torch.cat([img_feat.detach(), num_feat.detach()], dim=1)
            alpha = self.gate_network(gate_input)

            # c. 加权融合两个“被增强后”的、处于同等地位的特征
            f_all = alpha * attended_img_feat + (1 - alpha) * attended_num_feat
        
        # 简单拼接融合 (concat) 或单模态情况
        else:
            feats = []
            if img_feat is not None: feats.append(img_feat)
            if num_feat is not None: feats.append(num_feat)
            if not feats: raise ValueError("At least one branch must be enabled")
            f_all = torch.cat(feats, dim=1)
            
            # 为了兼容性，在单模态时也定义alpha，以便分析脚本统一处理
            if self.use_image and not self.use_pollution:
                alpha = torch.ones(B, 1, device=f_all.device)
            elif not self.use_image and self.use_pollution:
                alpha = torch.zeros(B, 1, device=f_all.device)

        # --- 3. 回归预测 ---
        output = self.regressor(f_all)

        # --- 4. 根据不同模式返回所需输出 ---
        # 训练时，如果使用对比学习，则额外返回特征
        if self.training and self.config.get("use_contrastive", False):
            if img_feat is None or num_feat is None:
                raise ValueError("Contrastive learning requires both image and pollution branches to be enabled.")
            return output, img_feat, num_feat
        
        # 推理时，如果需要贡献度，则额外返回alpha
        if return_contribution:
            if alpha is None:
                # 如果是concat融合且双模态，alpha未定义，返回一个代表“未定义”的-1值
                alpha = torch.full((B, 1), -1.0, device=f_all.device)
            return output, alpha 

        return output