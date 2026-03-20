import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



class ProjectedKLAlignment(nn.Module):
    """
    多模态 KL 对齐模块：
    - 将图像和数值特征分别映射到相同的低维空间（proj_dim）
    - 使用 softmax + KL 散度对其模态分布

    参数:
        img_dim: 图像特征原始维度（如 128）
        num_dim: 数值特征原始维度（如 256）
        proj_dim: 映射后的共享空间维度（如 64）
        reduction: KL 损失的计算方式（'batchmean' 推荐）
        detach_target: 是否对目标模态进行 detach（用于蒸馏式对齐）
    """
    def __init__(self, img_dim=128, num_dim=256, proj_dim=64,
                 reduction='batchmean', detach_target=True):
        super(ProjectedKLAlignment, self).__init__()
        self.proj_img = nn.Linear(img_dim, proj_dim)
        self.proj_num = nn.Linear(num_dim, proj_dim)
        self.reduction = reduction
        self.detach_target = detach_target

    def forward(self, img_feat, num_feat):
        # 1. 映射到共享对齐空间
        img_proj = self.proj_img(img_feat)  # [B, 64]
        num_proj = self.proj_num(num_feat)  # [B, 64]

        # 2. softmax -> 概率分布
        log_p = F.log_softmax(img_proj, dim=-1)
        q = F.softmax(num_proj.detach() if self.detach_target else num_proj, dim=-1)

        # 3. KL 散度
        kl_loss = F.kl_div(log_p, q, reduction=self.reduction)
        return kl_loss
