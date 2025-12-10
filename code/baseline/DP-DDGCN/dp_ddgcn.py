import os
import pickle
import random
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# 1. 工具类与函数 (Utilities)
# =============================================================================

def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """早停机制，防止过拟合"""
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
# 2. 图构建工具 (Graph Utilities)
# =============================================================================

def calculate_static_distance_matrix(station_coords_list, sigma=10, epsilon=0.5):
    """根据站点坐标计算静态距离邻接矩阵 (W_dis)"""
    num_stations = len(station_coords_list)
    dist_mx = np.zeros((num_stations, num_stations), dtype=np.float32)
    for i in range(num_stations):
        for j in range(i, num_stations):
            lon1, lat1 = station_coords_list[i]
            lon2, lat2 = station_coords_list[j]
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(np.radians(dlat/2))**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(dlon/2))**2
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            dist_mx[i, j] = dist
            dist_mx[j, i] = dist
    
    dists = dist_mx[~np.isinf(dist_mx)].flatten()
    std = dists.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < epsilon] = 0
    return torch.tensor(adj_mx, dtype=torch.float32)

def calculate_dynamic_wind_matrix(station_coords_tensor, wind_speed, wind_dir):
    """
    在单个时间步，根据风速和风向计算动态有向图 (W_dyn)
    - station_coords_tensor: [N, 2] (lon, lat)
    - wind_speed: [B, N]
    - wind_dir: [B, N]
    返回: [B, N, N]
    """
    B, N = wind_speed.shape
    
    station_vecs = station_coords_tensor.unsqueeze(0) - station_coords_tensor.unsqueeze(1)
    station_angles = torch.atan2(station_vecs[:, :, 1], station_vecs[:, :, 0])
    station_angles = station_angles.unsqueeze(0).expand(B, -1, -1).to(wind_speed.device)

    wind_rad = torch.deg2rad(wind_dir)
    wind_rad_expanded = wind_rad.unsqueeze(2).expand(-1, -1, N)

    angle_diff = station_angles - wind_rad_expanded
    cos_theta = torch.cos(angle_diff)
    
    wind_speed_expanded = wind_speed.unsqueeze(2).expand(-1, -1, N)
    W_dyn = wind_speed_expanded * cos_theta
    W_dyn[W_dyn < 0] = 0
    
    return W_dyn

# =============================================================================
# 3. 数据集定义 (Dataset Definition)
# =============================================================================

class AirQualityDataset(Dataset):
    def __init__(self, pkl_file, config):
        self.config = config
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history_len = self.config["history_len"]

        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32).permute(1, 0, 2)
        pollution_seq = pollution_seq[-history_len:, :, :]

        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)
        weather_seq = weather_seq[-history_len:, :]

        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        return pollution_seq, weather_seq, target

# =============================================================================
# 4. DP-DDGCN 模型定义 (Model Definition)
# =============================================================================

class GraphConv(nn.Module):
    """有向图卷积层 (基于随机游走)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        N = x.shape[1]
        W_tilde = adj + torch.eye(N, device=x.device).unsqueeze(0)
        D_tilde_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(W_tilde.sum(dim=2)) + 1e-6))
        norm_adj = torch.bmm(torch.bmm(D_tilde_inv_sqrt, W_tilde), D_tilde_inv_sqrt)
        output = torch.bmm(norm_adj, x)
        return self.linear(output)

class DP_DDGCB(nn.Module):
    """双路径动态有向图卷积块"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gcn_outgoing = GraphConv(in_features, out_features)
        self.gcn_incoming = GraphConv(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, W_out, W_in):
        h = self.relu(self.gcn_outgoing(x, W_out))
        output = self.relu(self.gcn_incoming(h, W_in))
        return output

class DP_DDGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_stations = config["num_stations"]
        self.pollutant_features = config["pollutant_features"]
        self.hidden_dim = config["hidden_dim"]
        self.pred_len = config["pred_len"]

        self.register_buffer('static_adj', calculate_static_distance_matrix(config["station_coords_list"]))
        self.register_buffer('station_coords_tensor', torch.tensor(config["station_coords_list"], dtype=torch.float32))

        self.dp_ddgcb = DP_DDGCB(self.pollutant_features, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.pollutant_features)
        )

    def forward(self, history_pollution, history_weather):
        B, T_h, N, F_p = history_pollution.shape
        device = history_pollution.device
        
        h = torch.zeros(1, B * N, self.hidden_dim, device=device)

        for t in range(T_h):
            x_t = history_pollution[:, t, :, :]
            weather_t = history_weather[:, t, :]

            wind_speed_scalar = weather_t[:, 2]
            wind_dir_scalar = weather_t[:, 1]
            
            wind_speed_t = wind_speed_scalar.unsqueeze(1).expand(-1, N)
            wind_dir_t = wind_dir_scalar.unsqueeze(1).expand(-1, N)

            W_dyn_t = calculate_dynamic_wind_matrix(self.station_coords_tensor, wind_speed_t, wind_dir_t)
            W_out_t = self.static_adj.unsqueeze(0) * W_dyn_t
            W_in_t = W_out_t.transpose(1, 2)
            
            spatial_features = self.dp_ddgcb(x_t, W_out_t, W_in_t)
            
            gru_input = spatial_features.reshape(B * N, 1, self.hidden_dim)
            _, h = self.gru(gru_input, h)

        last_W_out = W_out_t
        last_W_in = W_in_t
        
        outputs = []
        decoder_input = self.output_mlp(h.squeeze(0).view(B, N, -1))

        for _ in range(self.pred_len):
            spatial_features = self.dp_ddgcb(decoder_input, last_W_out, last_W_in)
            gru_input = spatial_features.reshape(B * N, 1, self.hidden_dim)
            _, h = self.gru(gru_input, h)
            
            prediction = self.output_mlp(h.squeeze(0).view(B, N, -1))
            outputs.append(prediction)
            decoder_input = prediction

        return torch.stack(outputs, dim=1)

# =============================================================================
# 5. 训练与评估主函数
# =============================================================================

def train_and_evaluate(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        if k != "station_coords_list":
             print(f"  {k}: {v}")

    dataset = AirQualityDataset(config["pkl_file"], config)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    print(f"\n数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    device = config["device"]
    model = DP_DDGCN(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型信息]")
    print(f"模型: {model.__class__.__name__}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=config.get("patience", 10))
    best_val_loss = float("inf")

    # ==================== 修正部分 START ====================
    # 确定用于近似目标站预测的邻居站点的索引
    # 'Wanshouxigong' 是邻居列表中的第一个
    NEIGHBOR_FOR_TARGET_IDX = 7
    # ==================== 修正部分 END ======================

    print("\n[开始训练]")
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for poll_seq, wea_seq, target in train_loader:
            poll_seq, wea_seq, target = poll_seq.to(device), wea_seq.to(device), target.to(device)
            
            pred_seq = model(poll_seq, wea_seq)
            
            # 适配：使用指定邻居站点的第一个预测步来近似目标站的预测
            pred_for_loss = pred_seq[:, 0, NEIGHBOR_FOR_TARGET_IDX, 0].unsqueeze(1)
            loss = loss_fn(pred_for_loss, target)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
            optimizer.step()
            total_train_loss += loss.item() * poll_seq.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for poll_seq, wea_seq, target in val_loader:
                poll_seq, wea_seq, target = poll_seq.to(device), wea_seq.to(device), target.to(device)
                pred_seq = model(poll_seq, wea_seq)
                pred_for_eval = pred_seq[:, 0, NEIGHBOR_FOR_TARGET_IDX, 0].unsqueeze(1)
                loss = loss_fn(pred_for_eval, target)
                total_val_loss += loss.item() * poll_seq.size(0)
                y_true.append(target.cpu().numpy())
                y_pred.append(pred_for_eval.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)

        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"[Epoch {epoch+1:03d}/{config['num_epochs']}] "
              f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
              f"R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['save_dir'], "dp_ddgcn_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> ✅ 模型已保存至 {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ 早停机制触发于 epoch {epoch+1}")
            break

    print("\n[开始测试]")
    best_model_path = os.path.join(config['save_dir'], "dp_ddgcn_best.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for poll_seq, wea_seq, target in test_loader:
            poll_seq, wea_seq, target = poll_seq.to(device), wea_seq.to(device), target.to(device)
            pred_seq = model(poll_seq, wea_seq)
            pred_for_eval = pred_seq[:, 0, NEIGHBOR_FOR_TARGET_IDX, 0].unsqueeze(1)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred_for_eval.cpu().numpy())
            
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n[测试结果]")
    print(f"  - R2  : {r2:.3f}")
    print(f"  - MAE : {mae:.3f}")
    print(f"  - RMSE: {rmse:.3f}")

    results = {"r2": r2, "mae": mae, "rmse": rmse}
    result_path = os.path.join(config['save_dir'], "dp_ddgcn_test_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 6. 主程序入口
# =============================================================================
STATION_COORDS = {
'dfmz': (121.4998, 31.2397),
'Shiwuchang': (121.567, 31.111),
'Hongkou': (121.4800, 31.2715),
'Shangshida': (121.4208, 31.1613),
'Yangpu': (121.5306, 31.2728),
'Qingpu': (121.1139, 31.1514),
'Jingan': (121.4456, 31.2230),
'PDchuansha': (121.6986, 31.1869),
'PDxinqu': (121.5508, 31.2105),
'PDzhangjiang': (121.5874, 31.2012),
}

NEIGHBOR_STATION_ORDER = [
    'Shiwuchang', 'Hongkou', 'Shangshida', 'Yangpu', 'Qingpu', 'Jingan',
    'PDchuansha', 'PDxinqu', 'PDzhangjiang'
]

DP_DDGCN_CONFIG = {
    "name": "DP_DDGCN_baseline",
    "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 3407,
    "save_dir": "./checkpoints/DP_DDGCN_baseline",

    "num_stations": 9, 
    "pollutant_features": 6,
    "station_coords_list": [STATION_COORDS[name] for name in NEIGHBOR_STATION_ORDER], # <-- 只使用邻居坐标

    "history_len": 6,
    "pred_len": 1,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 300,
    "patience": 20,
}
# ==================== 修正部分 END ======================
if __name__ == '__main__':
    # STATION_COORDS = {
    #     'bjdst': (116.300, 39.917), 'Wanshouxigong': (116.352, 39.878),
    #     'Dingling': (116.22, 40.292), 'Dongsi': (116.417, 39.929),
    #     'Tiantan': (116.407, 39.886), 'Nongzhanguan': (116.461, 39.937),
    #     'Guanyuan': (116.339, 39.929), 'Haidingquwanliu': (116.287, 39.987),
    #     'Shunyixincheng': (116.655, 40.127), 'Huairouzhen': (116.628, 40.328),
    #     'Changpingzhen': (116.23, 40.217), 'Aotizhongxin': (116.397, 39.982),
    #     'Gucheng': (116.184, 39.914)
    # }
    
    # # ==================== 修正部分 START ====================
    # # 错误根源是数据中只有12个站，而坐标有13个。
    # # 我们现在只使用12个邻居站点的坐标来构建图。
    # NEIGHBOR_STATION_ORDER = [
    #     'Wanshouxigong', 'Dingling', 'Dongsi', 'Tiantan', 
    #     'Nongzhanguan', 'Guanyuan', 'Haidingquwanliu', 'Shunyixincheng', 
    #     'Huairouzhen', 'Changpingzhen', 'Aotizhongxin', 'Gucheng'
    # ]



    os.makedirs(DP_DDGCN_CONFIG["save_dir"], exist_ok=True)
    train_and_evaluate(DP_DDGCN_CONFIG)