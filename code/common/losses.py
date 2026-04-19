import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, img_dim, num_dim, proj_dim=128, init_temp=0.1):
        super().__init__()
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

    def forward(self, img_feat, num_feat):
        """
        img_feat: [B, D_img]
        num_feat: [B, D_num]
        """
        temp = torch.exp(self.log_temp)

        img_proj = self.img_proj(img_feat)
        num_proj = self.num_proj(num_feat)

        img_proj = F.normalize(img_proj, dim=1)
        num_proj = F.normalize(num_proj, dim=1)

        logits = torch.mm(img_proj, num_proj.T) / temp
        labels = torch.arange(img_proj.size(0), device=img_proj.device)

        loss_i2n = F.cross_entropy(logits, labels)
        loss_n2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2n + loss_n2i) / 2

        return loss


class ContrastiveLossWithLabelThreshold(nn.Module):
    def __init__(self, img_dim, num_dim, proj_dim=64, init_temp=0.1, threshold=1):
        super().__init__()
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
        self.eps = 1e-12

    def forward(self, img_feat, num_feat, labels):
        temp = torch.exp(self.log_temp)
        labels = labels.squeeze(-1)

        img_proj = F.normalize(self.img_proj(img_feat), dim=1)
        num_proj = F.normalize(self.num_proj(num_feat), dim=1)

        logits = torch.mm(img_proj, num_proj.T) / temp

        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        pos_mask = (label_diff < self.threshold).float()
        pos_mask.fill_diagonal_(1)

        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)

        exp_logits = torch.exp(logits)
        neg_sum = (exp_logits * neg_mask).sum(1)
        safe_neg_sum = torch.clamp(neg_sum, min=self.eps)

        pos_term = (logits * pos_mask).sum(1)
        neg_term = torch.log(safe_neg_sum)
        contrastive_loss = - (pos_term - neg_term)

        if torch.isinf(contrastive_loss).any() or torch.isnan(contrastive_loss).any():
            return torch.tensor(0.0, device=img_feat.device, requires_grad=True)

        return contrastive_loss.mean()