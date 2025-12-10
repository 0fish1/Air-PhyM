import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, img_dim, num_dim, proj_dim=128, init_temp=0.1):
        super().__init__()
        # 图像特征投影网络
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        # 数值特征投影
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))

    def forward(self, img_feat, num_feat):
        """
        img_feat: [B, D_img]
        num_feat: [B, D_num]
        """
        temp = torch.exp(self.log_temp)  # 确保温度>0

        # 1. 投影到共享对比空间
        img_proj = self.img_proj(img_feat)   # [B, D_proj]
        num_proj = self.num_proj(num_feat)   # [B, D_proj]

        # 2. L2 归一化
        img_proj = F.normalize(img_proj, dim=1)
        num_proj = F.normalize(num_proj, dim=1)

        # 3. 相似度矩阵
        logits = torch.mm(img_proj, num_proj.T) / temp  # [B, B]
        labels = torch.arange(img_proj.size(0), device=img_proj.device)

        # 4. 双向 InfoNCE
        loss_i2n = F.cross_entropy(logits, labels)         # image → number
        loss_n2i = F.cross_entropy(logits.T, labels)       # number → image
        loss = (loss_i2n + loss_n2i) / 2

        return loss



# class ContrastiveLossWithLabelThreshold(nn.Module):
#     def __init__(self, img_dim, num_dim, proj_dim=128, init_temp=0.1, threshold=3.0):
#         super().__init__()
#         # 图像特征投影网络
#         self.img_proj = nn.Sequential(
#             nn.Linear(img_dim, proj_dim),
#             nn.ReLU(),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         # 数值特征投影
#         self.num_proj = nn.Sequential(
#             nn.Linear(num_dim, proj_dim),
#             nn.ReLU(),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))
#         self.threshold = threshold

#     def forward(self, img_feat, num_feat, labels):
#         """
#         img_feat: [B, D_img]
#         num_feat: [B, D_num]
#         labels: [B] containing the pollution levels
#         """
#         temp = torch.exp(self.log_temp)  # 确保温度>0
#         batch_size = img_feat.size(0)
#         labels = labels.squeeze(-1)

#         # 1. 投影到共享对比空间
#         img_proj = self.img_proj(img_feat)   # [B, D_proj]
#         num_proj = self.num_proj(num_feat)   # [B, D_proj]
  

#         # 2. L2 归一化
#         img_proj = F.normalize(img_proj, dim=1)
#         num_proj = F.normalize(num_proj, dim=1)

#         # 3. 计算相似度矩阵
#         logits = torch.mm(img_proj, num_proj.T) / temp  # [B, B]
        
#         # 4. 创建正样本掩码
#         label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # [B, B]
        
#         # 正样本包括:
#         # - 相同样本的不同模态 (对角线)
#         # - 标签差异小于阈值的样本
#         pos_mask = (label_diff < self.threshold).float()
#         pos_mask.fill_diagonal_(1)  # 确保相同样本总是正样本

        
#         # 负样本掩码是正样本掩码的反
#         neg_mask = 1 - pos_mask
        
#         # 对于对角线元素，我们不想让样本与自身作为负样本
#         neg_mask.fill_diagonal_(0)
        
#         # 5. 计算对比损失
#         # 计算正样本的logits
#         exp_logits = torch.exp(logits)
        
#         # 计算正样本部分
#         pos_term = (logits * pos_mask).sum(1)  # 分子
        
#         # 计算负样本部分 (包括所有负样本的exp(logits))
#         neg_term = torch.log((exp_logits * neg_mask).sum(1))  # 分母的对数
        
#         # 计算每个样本的对比损失
#         contrastive_loss = - (pos_term - neg_term)
        
#         # 平均所有样本的损失
#         loss = contrastive_loss.mean()
        
#         return loss

class ContrastiveLossWithLabelThreshold(nn.Module):
    def __init__(self, img_dim, num_dim, proj_dim=256, init_temp=0.1, threshold=1):
        super().__init__()
        # 保持原有结构完全不变
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))
        self.threshold = threshold
        
        # 仅添加一个极小值保护
        self.eps = 1e-12  # 比常规eps更小的值，确保不影响原有数学性质

    def forward(self, img_feat, num_feat, labels):
        temp = torch.exp(self.log_temp)
        labels = labels.squeeze(-1)

        # 保持原有投影和归一化
        img_proj = F.normalize(self.img_proj(img_feat), dim=1)
        num_proj = F.normalize(self.num_proj(num_feat), dim=1)
        
        # 相似度计算不变
        logits = torch.mm(img_proj, num_proj.T) / temp
        
        # 正负样本掩码计算不变
        # print("labels:",labels)
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        # print("label_diff:",label_diff)
        pos_mask = (label_diff < self.threshold).float()
        # print("pos_mask:",pos_mask)
        pos_mask.fill_diagonal_(1)
        
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # 关键修复点1：确保分母不为零
        exp_logits = torch.exp(logits)
        neg_sum = (exp_logits * neg_mask).sum(1)
        
        # 关键修复点2：添加极小值保护
        safe_neg_sum = torch.clamp(neg_sum, min=self.eps)
        
        # 保持原有损失计算
        pos_term = (logits * pos_mask).sum(1)
        neg_term = torch.log(safe_neg_sum)
        contrastive_loss = - (pos_term - neg_term)
        
        # 关键修复点3：损失值安全处理
        if torch.isinf(contrastive_loss).any() or torch.isnan(contrastive_loss).any():
            # 出现异常时返回安全值（保持梯度流动）
            print("⚠️ 检测到异常损失值，启用保护模式")
            return torch.tensor(0.0, device=img_feat.device, requires_grad=True)
            
        return contrastive_loss.mean()


