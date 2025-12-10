import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

# =============================================================================
# 2. 数据集定义与地理特征工程
# =============================================================================

class GeoMLPDataset(Dataset):
    def __init__(self, pkl_file, config, station_to_predict_idx):
        self.config = config
        self.station_to_predict_idx = station_to_predict_idx
        
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")

        # 预计算距离，用于S-PM2.5和DIS
        self.neighbor_coords = np.array(config["neighbor_station_coords"])
        self.target_coords = np.array(config["target_station_coord"])
        self.distances_to_target = np.array([
            haversine_distance(nc[0], nc[1], self.target_coords[0], self.target_coords[1])
            for nc in self.neighbor_coords
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # --- 基础特征 ---
        # 使用最新的气象数据
        weather_features = torch.tensor(sample['weather_seq'][-1, :], dtype=torch.float32)
        
        # --- 地理智能特征计算 ---
        neighbor_pollution_seq = sample['pollution_seq'] # Shape: [N_neighbors, T, F]
        
        # 1. S-PM2.5 (空间项)
        # 使用邻居站在当前时刻(最后一个时间步)的PM2.5值
        current_neighbor_pm25 = neighbor_pollution_seq[:, -1, 0]
        spatial_distances = self.distances_to_target
        spatial_weights = 1.0 / (spatial_distances**2 + 1e-6)
        s_pm25 = np.sum(spatial_weights * current_neighbor_pm25) / np.sum(spatial_weights)
        s_pm25 = torch.tensor([s_pm25], dtype=torch.float32)

        # 2. T-PM2.5 (时间项)
        # 适配：由于目标站是空白的，我们用离目标站最近的邻居站的历史来近似
        # 在您的数据中，Wanshouxigong是第一个邻居，且离bjdst最近
        closest_neighbor_history = neighbor_pollution_seq[0, :, 0]
        temporal_distances = np.arange(len(closest_neighbor_history), 0, -1)
        temporal_weights = 1.0 / (temporal_distances**2 + 1e-6)
        t_pm25 = np.sum(temporal_weights * closest_neighbor_history) / np.sum(temporal_weights)
        t_pm25 = torch.tensor([t_pm25], dtype=torch.float32)

        # 3. DIS (地理距离项)
        # 使用到最近邻居的距离
        dis = torch.tensor([np.min(spatial_distances)], dtype=torch.float32)

        # --- 拼接所有特征 ---
        # 论文输入: f(AOD, RH, WS, TMP, PBL, PS, NDVI, S-PM2.5, T-PM2.5, DIS)
        # 您的数据中 weather_seq 包含: T, P, RH, WindDir, WindSpeed
        # 我们拼接: Weather (5) + S-PM2.5 (1) + T-PM2.5 (1) + DIS (1) = 8 features
        input_features = torch.cat([weather_features, s_pm25, t_pm25, dis])
        
        # --- 目标值 ---
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        return input_features, target

# =============================================================================
# 3. Geo-MLP 模型定义
# =============================================================================

class GeoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

# =============================================================================
# 4. 训练与评估主函数
# =============================================================================

def train_and_evaluate(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        if "coords" not in k: print(f"  {k}: {v}")

    # --- 数据准备 ---
    # 在空间估计任务中，我们通常留出部分站点作为测试集
    # 这里我们遵循您的项目设定，留出目标站bjdst作为唯一的“空白”测试点
    dataset = GeoMLPDataset(config["pkl_file"], config, station_to_predict_idx=0)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    print(f"\n数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    # --- 模型、优化器、损失函数 ---
    device = config["device"]
    model = GeoMLP(input_dim=config["input_features"]).to(device)
    
    print(f"\n[模型信息] 模型: {model.__class__.__name__}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=config.get("patience", 10))
    best_val_loss = float("inf")

    # --- 训练与验证循环 ---
    print("\n[开始训练]")
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            pred = model(features)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * features.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(device), target.to(device)
                pred = model(features)
                loss = loss_fn(pred, target)
                total_val_loss += loss.item() * features.size(0)
        avg_val_loss = total_val_loss / len(val_dataset)

        print(f"[Epoch {epoch+1:03d}/{config['num_epochs']}] "
              f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['save_dir'], "geo_mlp_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> ✅ 模型已保存至 {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ 早停机制触发于 epoch {epoch+1}")
            break

    # --- 测试 ---
    print("\n[开始测试]")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], "geo_mlp_best.pth")))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            pred = model(features)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            
    y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n[Geo-MLP Baseline 测试结果]")
    print(f"  - R2  : {r2:.3f}\n  - MAE : {mae:.3f}\n  - RMSE: {rmse:.3f}")

    results = {"r2": r2, "mae": mae, "rmse": rmse}
    result_path = os.path.join(config['save_dir'], "geo_mlp_test_results.pkl")
    with open(result_path, 'wb') as f: pickle.dump(results, f)
    print(f"\n✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 5. 主程序入口
# =============================================================================

if __name__ == '__main__':
    STATION_COORDS = {
        'bjdst': (116.300, 39.917), 'Wanshouxigong': (116.352, 39.878),
        'Dingling': (116.22, 40.292), 'Dongsi': (116.417, 39.929),
        'Tiantan': (116.407, 39.886), 'Nongzhanguan': (116.461, 39.937),
        'Guanyuan': (116.339, 39.929), 'Haidingquwanliu': (116.287, 39.987),
        'Shunyixincheng': (116.655, 40.127), 'Huairouzhen': (116.628, 40.328),
        'Changpingzhen': (116.23, 40.217), 'Aotizhongxin': (116.397, 39.982),
        'Gucheng': (116.184, 39.914)
    }
    
    NEIGHBOR_STATION_ORDER = [
        'Wanshouxigong', 'Dingling', 'Dongsi', 'Tiantan', 'Nongzhanguan', 'Guanyuan', 
        'Haidingquwanliu', 'Shunyixincheng', 'Huairouzhen', 'Changpingzhen', 
        'Aotizhongxin', 'Gucheng'
    ]
    
    GEO_MLP_CONFIG = {
        "name": "Geo-MLP_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "./checkpoints/Geo-MLP_baseline",
        
        "target_station_coord": STATION_COORDS['bjdst'],
        "neighbor_station_coords": [STATION_COORDS[name] for name in NEIGHBOR_STATION_ORDER],
        
        "input_features": 8, # 5 weather + 3 geo-intelligent features
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_epochs": 150,
        "patience": 15,
    }

    os.makedirs(GEO_MLP_CONFIG["save_dir"], exist_ok=True)
    train_and_evaluate(GEO_MLP_CONFIG)