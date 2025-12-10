import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_squared_error

# ==============================
# Dataset
# ==============================
class AirQualityDataset(Dataset):
    def __init__(self, pkl_file, config):
        self.config = config
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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
        sample['pollution_seq'] = sample['pollution_seq'][:self.config["site_nums"], -self.config["history_hours"]:, :]
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

        # 气象序列 [T, F]
        sample['weather_seq'] = sample['weather_seq'][-self.config["history_hours"]:, :]
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

        # 目标值
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        return imgs, pollution_seq, weather_seq, target


# ==============================
# Collate Function
# ==============================
def collate_fn_samples(batch):
    imgs, pols, weas, targets = zip(*batch)
    imgs = torch.stack(imgs)
    pols = torch.stack(pols)
    weas = torch.stack(weas)
    targets = torch.stack(targets)
    return imgs, pols, weas, targets


# ==============================
# STGCN 模型
# ==============================
class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, (1, kernel_size), padding=(0, kernel_size // 2))

    def forward(self, x):
        # x: [B, C, N, T]
        return F.relu(self.conv(x))


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.temporal1 = TemporalConvLayer(c_in, c_out)
        self.spatial = nn.Conv2d(c_out, c_out, (1, 1))  # 简化的空间卷积
        self.temporal2 = TemporalConvLayer(c_out, c_out)

    def forward(self, x):
        x = self.temporal1(x)
        x = self.spatial(x)
        x = self.temporal2(x)
        return x


class STGCN(nn.Module):
    def __init__(self, c_in, c_out, k_top=3):
        super().__init__()
        self.block1 = STGCNBlock(c_in, 64)
        self.block2 = STGCNBlock(64, 64)
        self.fc = nn.Linear(64, c_out)
        self.k_top = k_top

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, F, N, T]

        x = self.block1(x)
        x = self.block2(x)

        x = x.mean(-1)  # [B, C, N]
        x = x.permute(0, 2, 1)  # [B, N, C]

        # Top-K 节点聚合 (预测目标站点)
        scores = x.mean(-1)  # [B, N]
        topk_val, topk_idx = torch.topk(scores, self.k_top, dim=-1)

        batch_out = []
        for i in range(B):
            nodes_feat = x[i, topk_idx[i], :]  # [K, C]
            agg = nodes_feat.mean(0)  # [C]
            out = self.fc(agg)
            batch_out.append(out)
        out = torch.stack(batch_out, dim=0)  # [B, c_out]
        return out


# ==============================
# 训练和验证
# ==============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, pols, weas, targets in loader:
        pols, targets = pols.to(device), targets.to(device)

        preds = model(pols)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pols.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for imgs, pols, weas, targets in loader:
            pols, targets = pols.to(device), targets.to(device)

            preds = model(pols)
            loss = criterion(preds, targets)
            total_loss += loss.item() * pols.size(0)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return total_loss / len(loader.dataset), preds_all, targets_all


# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    config = {
        "site_nums": 9,
        "history_hours": 24
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AirQualityDataset(
        "/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl",
        config
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn_samples)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn_samples)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, collate_fn=collate_fn_samples)

    print(f"Total: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")

    model = STGCN(c_in=6, c_out=1, k_top=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(30):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # 测试
    test_loss, preds, targets = evaluate(model, test_loader, criterion, device)
    mae = mean_absolute_error(targets, preds)
    # rmse = mean_squared_error(targets, preds, squared=False)
    mse = mean_squared_error(targets, preds)
    rmse = mse ** 0.5
    r2 = r2_score(targets, preds)

    print(f"Test MSE Loss={test_loss:.4f}")
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
