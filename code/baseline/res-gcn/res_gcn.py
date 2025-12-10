import os
import pickle
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 检查并提示安装 fastdtw
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("="*50)
    print("错误: 缺少 'fastdtw' 库。")
    print("请通过 pip 安装: pip install fastdtw")
    print("="*50)
    exit()

# =============================================================================
# 1. 工具类与函数 (Utilities)
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

# =============================================================================
# 2. 数据集定义 (Dataset Definition)
# =============================================================================

class AirQualityDataset(Dataset):
    def __init__(self, pkl_file, config):
        self.config = config
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history_len = self.config["history_len"]

        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)[:, -history_len:, :]
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)[-history_len:, :]
        
        num_stations = pollution_seq.shape[0]
        weather_seq_expanded = weather_seq.unsqueeze(0).expand(num_stations, -1, -1)
        
        history_data = torch.cat([pollution_seq, weather_seq_expanded], dim=-1).permute(1, 0, 2)

        img_path = sample['images'][-1] if sample['images'] else None
        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            else:
                image = torch.zeros(3, 224, 224)
        except Exception:
            image = torch.zeros(3, 224, 224)

        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        return history_data, image, target

# =============================================================================
# 3. Res-GCN 模型定义 (Model Definition)
# =============================================================================

# --- ResNet 分支 ---
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet_ImageExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResNetBasicBlock(64, 64, stride=1)
        self.layer2 = ResNetBasicBlock(64, 128, stride=2)
        self.layer3 = ResNetBasicBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# --- DSTGCN 分支 ---
def calculate_dtw_adj(batch_x, phi=1.0, epsilon=0.5):
    B, T, N, F = batch_x.shape
    adj = torch.zeros(B, N, N, device=batch_x.device)
    
    pm25_series_tensor = batch_x[:, :, :, 0].permute(0, 2, 1).unsqueeze(-1)
    pm25_series = pm25_series_tensor.cpu().numpy()

    for b in range(B):
        for i in range(N):
            for j in range(i, N):
                dist, _ = fastdtw(pm25_series[b, i], pm25_series[b, j], dist=euclidean)
                weight = np.exp(- (dist**2) / phi)
                if weight >= epsilon:
                    adj[b, i, j] = weight
                    adj[b, j, i] = weight
    return adj

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        adj_tilde = adj + torch.eye(adj.shape[1], device=x.device).unsqueeze(0)
        deg = torch.sum(adj_tilde, dim=2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm_adj = torch.einsum('bi,bij,bj->bij', deg_inv_sqrt, adj_tilde, deg_inv_sqrt)
        output = torch.bmm(norm_adj, x)
        return self.linear(output)

# ==================== 修正部分 START ====================
class Chomp1d(nn.Module):
    """用于裁剪TCN中多余padding的模块"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """修正后的TCN残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + res)
# ==================== 修正部分 END ======================

class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, in_dim)
        self.W2 = nn.Linear(in_dim, in_dim)
        self.V = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.V(torch.tanh(self.W1(x) + self.W2(x)))
        attention_weights = torch.softmax(scores, dim=1)
        return torch.sum(x * attention_weights, dim=1)

class STConvBlock(nn.Module):
    def __init__(self, in_features, gcn_dim, tcn_dim):
        super().__init__()
        self.gcn = GraphConvLayer(in_features, gcn_dim)
        self.tcn = TCNBlock(gcn_dim, tcn_dim)
        self.attention = TemporalAttention(tcn_dim)
        self.ln = nn.LayerNorm(tcn_dim)

    def forward(self, x, adj):
        B, T, N, F = x.shape
        gcn_out_list = [self.gcn(x[:, t, :, :], adj) for t in range(T)]
        gcn_out = torch.stack(gcn_out_list, dim=1)
        
        gcn_out_reshaped = gcn_out.permute(0, 2, 3, 1).reshape(B*N, -1, T)
        tcn_out = self.tcn(gcn_out_reshaped).reshape(B, N, -1, T).permute(0, 3, 1, 2)
        
        attn_out_list = [self.attention(tcn_out[:, :, n, :]) for n in range(N)]
        attn_out = torch.stack(attn_out_list, dim=1)
        
        return self.ln(attn_out)

class DSTGCN_TimeSeriesExtractor(nn.Module):
    def __init__(self, in_features, out_dim):
        super().__init__()
        self.st_block1 = STConvBlock(in_features, 64, 128)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        adj = calculate_dtw_adj(x)
        h1 = self.st_block1(x, adj)
        return self.fc(h1)

# --- Main Res-GCN Model ---
class Res_GCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_stations = config["num_stations"]
        self.pred_len = config["pred_len"]
        
        self.image_branch = ResNet_ImageExtractor(output_dim=config["hidden_dim"])
        self.ts_branch = DSTGCN_TimeSeriesExtractor(in_features=config["input_features"], out_dim=config["hidden_dim"])
        
        self.W_st = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.W_v = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.bias_z = nn.Parameter(torch.zeros(config["hidden_dim"]))
        
        self.conv1d = nn.Conv1d(self.num_stations, self.pred_len, kernel_size=1)
        self.fc = nn.Linear(config["hidden_dim"], 1)

    def forward(self, history_data, image):
        h_v = self.image_branch(image)
        h_st = self.ts_branch(history_data)
        
        h_v_expanded = h_v.unsqueeze(1).expand(-1, self.num_stations, -1)
        z = torch.sigmoid(self.W_st(h_st) + self.W_v(h_v_expanded) + self.bias_z)
        h_f = z * h_st + (1 - z) * h_v_expanded
        
        out = self.conv1d(h_f)
        out = self.fc(out)
        
        return out.squeeze(-1)

# =============================================================================
# 4. 训练与评估主函数 (保持不变)
# =============================================================================

def train_and_evaluate(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        if k != "station_coords_list": print(f"  {k}: {v}")

    dataset = AirQualityDataset(config["pkl_file"], config)
    train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    print(f"\n数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    device = config["device"]
    model = Res_GCN(config).to(device)
    
    print(f"\n[模型信息] 模型: {model.__class__.__name__}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=config.get("patience", 10))
    best_val_loss = float("inf")

    print("\n[开始训练]")
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for hist_data, image, target in train_loader:
            hist_data, image, target = hist_data.to(device), image.to(device), target.to(device)
            
            pred_seq = model(hist_data, image)
            pred_for_loss = pred_seq[:, 0].unsqueeze(1)
            loss = loss_fn(pred_for_loss, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * hist_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for hist_data, image, target in val_loader:
                hist_data, image, target = hist_data.to(device), image.to(device), target.to(device)
                pred_seq = model(hist_data, image)
                pred_for_eval = pred_seq[:, 0].unsqueeze(1)
                loss = loss_fn(pred_for_eval, target)
                total_val_loss += loss.item() * hist_data.size(0)
                y_true.append(target.cpu().numpy())
                y_pred.append(pred_for_eval.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)

        y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()
        r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"[Epoch {epoch+1:03d}/{config['num_epochs']}] "
              f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
              f"R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['save_dir'], "res_gcn_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> ✅ 模型已保存至 {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ 早停机制触发于 epoch {epoch+1}")
            break

    print("\n[开始测试]")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], "res_gcn_best.pth")))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for hist_data, image, target in test_loader:
            hist_data, image, target = hist_data.to(device), image.to(device), target.to(device)
            pred_seq = model(hist_data, image)
            pred_for_eval = pred_seq[:, 0].unsqueeze(1)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred_for_eval.cpu().numpy())
            
    y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()
    r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n[测试结果]")
    print(f"  - R2  : {r2:.3f}\n  - MAE : {mae:.3f}\n  - RMSE: {rmse:.3f}")

    results = {"r2": r2, "mae": mae, "rmse": rmse}
    result_path = os.path.join(config['save_dir'], "res_gcn_test_results.pkl")
    with open(result_path, 'wb') as f: pickle.dump(results, f)
    print(f"\n✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 5. 主程序入口 (保持不变)
# =============================================================================
RES_GCN_CONFIG = {
        "name": "Res_GCN_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "./checkpoints/Res_GCN_baseline",
        
        "num_stations": 12,
        "input_features": 11,
        
        "history_len": 24,
        "pred_len": 1,
        "hidden_dim": 4,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 100,
        "patience": 10,
    }
if __name__ == '__main__':
    

    os.makedirs(RES_GCN_CONFIG["save_dir"], exist_ok=True)
    train_and_evaluate(RES_GCN_CONFIG)