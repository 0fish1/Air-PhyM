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
    def __init__(self, img_dim=128, num_dim=256, proj_dim=64,
                 reduction='batchmean', detach_target=True):
        super(ProjectedKLAlignment, self).__init__()
        self.proj_img = nn.Linear(img_dim, proj_dim)
        self.proj_num = nn.Linear(num_dim, proj_dim)
        self.reduction = reduction
        self.detach_target = detach_target

    def forward(self, img_feat, num_feat):
        img_proj = self.proj_img(img_feat)
        num_proj = self.proj_num(num_feat)

        log_p = F.log_softmax(img_proj, dim=-1)
        q = F.softmax(num_proj.detach() if self.detach_target else num_proj, dim=-1)

        kl_loss = F.kl_div(log_p, q, reduction=self.reduction)
        return kl_loss