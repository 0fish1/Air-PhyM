import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def collate_fn_samples(batch):
    imgs, pols, weas, targets = zip(*batch)
    imgs = torch.stack(imgs)
    pols = torch.stack(pols)
    weas = torch.stack(weas)
    targets = torch.stack(targets)
    return imgs, pols, weas, targets


# ==============================
# Graph Convolution Layer
# ==============================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # x: [B, N, F]
        # adj: [N, N]
        support = self.fc(x)  # [B, N, out_features]
        out = torch.bmm(adj.unsqueeze(0).repeat(x.size(0), 1, 1), support)
        return out


# ==============================
# GC-RNN 模型
# ==============================
class GCRNN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_dim, adj, rnn_type="GRU"):
        super(GCRNN, self).__init__()
        self.adj = adj
        self.gc = GraphConvolution(in_features, hidden_dim)
        self.rnn_type = rnn_type

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape

        # 时序展开
        x = x.permute(0, 2, 1, 3)  # [B, T, N, F]
        outs = []
        for t in range(T):
            xt = x[:, t, :, :]  # [B, N, F]
            xt = self.gc(xt, self.adj)  # [B, N, hidden]
            xt = xt.mean(1)  # 全局聚合: [B, hidden]
            outs.append(xt.unsqueeze(1))  # [B, 1, hidden]

        outs = torch.cat(outs, dim=1)  # [B, T, hidden]

        
        # 处理RNN输出
        if self.rnn_type == "GRU":
            _, h = self.rnn(outs)  # h: [1, B, hidden]
            h = h.squeeze(0)  # [B, hidden]
        else:  # LSTM
            _, (h, _) = self.rnn(outs)  # h: [1, B, hidden]
            h = h.squeeze(0)  # [B, hidden]
        

        out = self.fc(h)  # [B, out_dim]
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
        "site_nums": 12,
        "history_hours": 24
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AirQualityDataset(
        "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
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

    # 构造邻接矩阵 (这里用全连接 + 自环)
    N = config["site_nums"]
    adj = torch.ones(N, N)
    adj = adj / adj.sum(1, keepdim=True)  # 行归一化
    adj = adj.to(device)

    model = GCRNN(num_nodes=N, in_features=6, hidden_dim=128, out_dim=1, adj=adj).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(300):
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
