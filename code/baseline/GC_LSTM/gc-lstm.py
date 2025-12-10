# import os
# import pickle
# import random
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # =============================================================================
# # 1. 工具类与函数 (Utilities)
# # =============================================================================

# def set_seed(seed=42):
#     """设置随机种子以确保实验可复现"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# class EarlyStopping:
#     """早停机制，防止过拟合"""
#     def __init__(self, patience=10, delta=0.0):
#         self.patience = patience
#         self.delta = delta
#         self.best_loss = float('inf')
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if val_loss < self.best_loss - self.delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True

# # =============================================================================
# # 2. 图构建工具 (Graph Utilities)
# # =============================================================================

# def calculate_gclstm_static_adj(station_coords_list, threshold=200.0):
#     """根据逆距离加权法计算 GC-LSTM 的静态邻接矩阵"""
#     num_stations = len(station_coords_list)
#     adj_mx = np.zeros((num_stations, num_stations), dtype=np.float32)
    
#     for i in range(num_stations):
#         for j in range(i, num_stations):
#             if i == j:
#                 continue
#             lon1, lat1 = station_coords_list[i]
#             lon2, lat2 = station_coords_list[j]
#             dlon, dlat = lon2 - lon1, lat2 - lat1
#             a = np.sin(np.radians(dlat/2))**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(dlon/2))**2
#             dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))

#             if dist < threshold:
#                 adj_mx[i, j] = 1.0 / dist
#                 adj_mx[j, i] = 1.0 / dist
                
#     return torch.tensor(adj_mx, dtype=torch.float32)

# def get_laplacian(adj):
#     """计算归一化的拉普拉斯矩阵 L = I - D^(-1/2) * A * D^(-1/2)"""
#     adj_tilde = adj + torch.eye(adj.shape[0], device=adj.device)
#     deg = torch.sum(adj_tilde, dim=1)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
#     laplacian = torch.eye(adj.shape[0], device=adj.device) - \
#                 torch.mm(torch.mm(torch.diag(deg_inv_sqrt), adj_tilde), torch.diag(deg_inv_sqrt))
#     return laplacian

# # =============================================================================
# # 3. 数据集定义 (Dataset Definition)
# # =============================================================================

# class AirQualityDataset(Dataset):
#     def __init__(self, pkl_file, config):
#         self.config = config
#         print(f"正在从 {pkl_file} 加载数据...")
#         with open(pkl_file, "rb") as f:
#             self.samples = pickle.load(f)
#         print("数据加载完成。")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         history_len = self.config["history_len"]

#         # 历史污染物数据 [sites, time, features]
#         pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)
#         pollution_seq = pollution_seq[:, -history_len:, :]

#         # 历史气象数据 [time, features]
#         weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)
#         weather_seq = weather_seq[-history_len:, :]
        
#         # 将全局气象数据广播到每个站点
#         num_stations = pollution_seq.shape[0]
#         weather_seq_expanded = weather_seq.unsqueeze(0).expand(num_stations, -1, -1)
        
#         # 拼接污染物和气象特征
#         # 最终形状: [time, sites, features]
#         history_data = torch.cat([pollution_seq, weather_seq_expanded], dim=-1).permute(1, 0, 2)

#         # 目标值
#         target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

#         return history_data, target

# # =============================================================================
# # 4. GC-LSTM 模型定义 (Model Definition)
# # =============================================================================

# class ChebConvLayer(nn.Module):
#     """自定义Chebyshev图卷积层"""
#     def __init__(self, in_features, out_features, K):
#         super().__init__()
#         self.K = K
#         self.linear = nn.Linear(in_features * K, out_features)

#     def forward(self, x, laplacian):
#         # x: [B, N, F_in]
#         B, N, F_in = x.shape
        
#         x0 = x
#         cheb_polynomials = [x0]
#         if self.K > 1:
#             x1 = torch.bmm(laplacian.expand(B, -1, -1), x)
#             cheb_polynomials.append(x1)
#         for k in range(2, self.K):
#             xn = 2 * torch.bmm(laplacian.expand(B, -1, -1), cheb_polynomials[-1]) - cheb_polynomials[-2]
#             cheb_polynomials.append(xn)
            
#         x_cheb = torch.cat(cheb_polynomials, dim=-1)
#         return self.linear(x_cheb)

# class GC_LSTM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.num_stations = config["num_stations"]
#         self.input_features = config["input_features"]
#         self.hidden_dim = config["hidden_dim"]
        
#         adj = calculate_gclstm_static_adj(config["station_coords_list"])
#         laplacian = get_laplacian(adj)
#         self.register_buffer('laplacian', laplacian)

#         self.gcn = ChebConvLayer(self.input_features, self.hidden_dim, K=2)
        
#         self.lstm = nn.LSTM(
#             input_size=self.input_features + self.hidden_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
        
#         self.regressor = nn.Linear(self.hidden_dim, 1)

#     def forward(self, history_data):
#         B, T_h, N, F_in = history_data.shape
        
#         lstm_inputs = []
#         for t in range(T_h):
#             x_t = history_data[:, t, :, :]
#             h_t = self.gcn(x_t, self.laplacian)
#             lstm_input_t = torch.cat([x_t, h_t], dim=-1)
#             lstm_inputs.append(lstm_input_t)
            
#         lstm_inputs = torch.stack(lstm_inputs, dim=1)
        
#         lstm_inputs_reshaped = lstm_inputs.view(B * N, T_h, -1)
#         lstm_outputs, _ = self.lstm(lstm_inputs_reshaped)
        
#         last_step_output = lstm_outputs[:, -1, :]
        
#         prediction = self.regressor(last_step_output)
        
#         return prediction.view(B, N, 1)

# # =============================================================================
# # 5. 训练与评估主函数
# # =============================================================================

# def train_and_evaluate(config):
#     set_seed(config["seed"])
    
#     print("\n[配置信息]")
#     for k, v in config.items():
#         if k != "station_coords_list":
#              print(f"  {k}: {v}")

#     dataset = AirQualityDataset(config["pkl_file"], config)
#     train_size = int(0.7 * len(dataset))
#     val_size = int(0.15 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
#     print(f"\n数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

#     device = config["device"]
#     model = GC_LSTM(config).to(device)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n[模型信息]")
#     print(f"模型: {model.__class__.__name__}")
#     print(f"总参数数量: {total_params:,}")
#     print(f"可训练参数数量: {trainable_params:,}")
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
#     loss_fn = nn.MSELoss()
#     early_stopper = EarlyStopping(patience=config.get("patience", 10))
#     best_val_loss = float("inf")

#     NEIGHBOR_FOR_TARGET_IDX = 6  # 目标站点在邻居列表中的索引位置

#     print("\n[开始训练]")
#     for epoch in range(config["num_epochs"]):
#         model.train()
#         total_train_loss = 0
#         for history_data, target in train_loader:
#             history_data, target = history_data.to(device), target.to(device)
            
#             pred = model(history_data)
            
#             pred_for_loss = pred[:, NEIGHBOR_FOR_TARGET_IDX, :]
#             loss = loss_fn(pred_for_loss, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item() * history_data.size(0)
#         avg_train_loss = total_train_loss / len(train_dataset)

#         model.eval()
#         total_val_loss = 0
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for history_data, target in val_loader:
#                 history_data, target = history_data.to(device), target.to(device)
#                 pred = model(history_data)
#                 pred_for_eval = pred[:, NEIGHBOR_FOR_TARGET_IDX, :]
#                 loss = loss_fn(pred_for_eval, target)
#                 total_val_loss += loss.item() * history_data.size(0)
#                 y_true.append(target.cpu().numpy())
#                 y_pred.append(pred_for_eval.cpu().numpy())
#         avg_val_loss = total_val_loss / len(val_dataset)

#         y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()
#         r2 = r2_score(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))

#         print(f"[Epoch {epoch+1:03d}/{config['num_epochs']}] "
#               f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
#               f"R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             save_path = os.path.join(config['save_dir'], "gc_lstm_best.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"  -> ✅ 模型已保存至 {save_path}")

#         early_stopper(avg_val_loss)
#         if early_stopper.early_stop:
#             print(f"⏹️ 早停机制触发于 epoch {epoch+1}")
#             break

#     print("\n[开始测试]")
#     best_model_path = os.path.join(config['save_dir'], "gc_lstm_best.pth")
#     model.load_state_dict(torch.load(best_model_path))
#     model.eval()
    
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for history_data, target in test_loader:
#             history_data, target = history_data.to(device), target.to(device)
#             pred = model(history_data)
#             pred_for_eval = pred[:, NEIGHBOR_FOR_TARGET_IDX, :]
#             y_true.append(target.cpu().numpy())
#             y_pred.append(pred_for_eval.cpu().numpy())
            
#     y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()
#     r2 = r2_score(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))

#     print("\n[测试结果]")
#     print(f"  - R2  : {r2:.3f}")
#     print(f"  - MAE : {mae:.3f}")
#     print(f"  - RMSE: {rmse:.3f}")

#     results = {"r2": r2, "mae": mae, "rmse": rmse}
#     result_path = os.path.join(config['save_dir'], "gc_lstm_test_results.pkl")
#     with open(result_path, 'wb') as f:
#         pickle.dump(results, f)
#     print(f"✅ 测试结果已保存至 {result_path}")

# # =============================================================================
# # 6. 主程序入口
# # =============================================================================

# if __name__ == '__main__':
#     STATION_COORDS = {
#         'bjdst': (116.300, 39.917), 'Wanshouxigong': (116.352, 39.878),
#         'Dingling': (116.22, 40.292), 'Dongsi': (116.417, 39.929),
#         'Tiantan': (116.407, 39.886), 'Nongzhanguan': (116.461, 39.937),
#         'Guanyuan': (116.339, 39.929), 'Haidingquwanliu': (116.287, 39.987),
#         'Shunyixincheng': (116.655, 40.127), 'Huairouzhen': (116.628, 40.328),
#         'Changpingzhen': (116.23, 40.217), 'Aotizhongxin': (116.397, 39.982),
#         'Gucheng': (116.184, 39.914)
#     }
    
#     NEIGHBOR_STATION_ORDER = [
#         'Wanshouxigong', 'Dingling', 'Dongsi', 'Tiantan', 
#         'Nongzhanguan', 'Guanyuan', 'Haidingquwanliu', 'Shunyixincheng', 
#         'Huairouzhen', 'Changpingzhen', 'Aotizhongxin', 'Gucheng'
#     ]
    
#     GC_LSTM_CONFIG = {
#         "name": "GC_LSTM_baseline",
#         "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
#         "device": "cuda:0" if torch.cuda.is_available() else "cpu",
#         "seed": 3407,
#         "save_dir": "./checkpoints/GC_LSTM_baseline",
        
#         "num_stations": 12,
#         "pollutant_features": 6,
#         "weather_features": 5,
#         "input_features": 11, # 6 (pollutants) + 5 (weather)
#         "station_coords_list": [STATION_COORDS[name] for name in NEIGHBOR_STATION_ORDER],

#         "history_len": 24,      # 论文中使用24h，这里适配48h数据，取36h=12, 48h=16
#         "hidden_dim": 64,       # GCN和LSTM的隐藏维度
#         "learning_rate": 1e-3,
#         "batch_size": 64,
#         "num_epochs": 150,
#         "patience": 20,
#     }

#     os.makedirs(GC_LSTM_CONFIG["save_dir"], exist_ok=True)
#     train_and_evaluate(GC_LSTM_CONFIG)





import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sin, cos, sqrt, asin, radians
from sklearn.metrics import mean_absolute_error, mean_squared_error

# EarlyStopping class
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

# Haversine function
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# ChebGCN layer
class ChebGCN(nn.Module):
    def __init__(self, in_feats, out_feats, K):
        super(ChebGCN, self).__init__()
        self.K = K
        self.theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_feats, out_feats)) for _ in range(K + 1)])

    def forward(self, X, cheb_polys):
        # X: [B, N, in_feats]
        # cheb_polys: list of [N, N] tensors
        out = 0
        for k in range(self.K + 1):
            t_x = torch.matmul(X, self.theta[k])  # [B, N, out_feats]
            out += torch.matmul(cheb_polys[k], t_x)  # [B, N, out_feats]
        return torch.relu(out)

# GC-LSTM Model
class GCLSTM(nn.Module):
    def __init__(self, config):
        super(GCLSTM, self).__init__()
        self.config = config
        self.device = config['device']
        self.N = 13  # 1 target + 12 neighbors
        self.F = 6 + 5 + 2  # aq + meteo + latlon
        self.gcn_out = 128
        self.lstm_hidden = 128
        self.K = 3  # Chebyshev order
        self.history_hours = config['history_hours']

        # Get coords
        station_coords_dict = config["station_coords"]
        station_list = ['bjdst'] + sorted(k for k in station_coords_dict if k != 'bjdst')
        coords = np.array([station_coords_dict[s] for s in station_list])  # [N, 2] (lon, lat)

        # Compute A
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])  # lon, lat
                    if d < 200 and d > 0:
                        A[i, j] = 1 / d
        A = (A + A.T) / 2  # Symmetric

        # Compute Laplacian
        D = np.diag(np.sum(A, axis=1))
        D_sqrt_inv = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-6))
        L = np.eye(self.N) - D_sqrt_inv @ A @ D_sqrt_inv

        # lambda_max
        eigenvalues = np.linalg.eigvalsh(L)
        lambda_max = eigenvalues.max()

        # Scaled L_tilde
        L_tilde = (2 / lambda_max) * L - np.eye(self.N)
        L_tilde = torch.from_numpy(L_tilde).float().to(self.device)

        # Chebyshev polynomials
        cheb_polys = [torch.eye(self.N).to(self.device), L_tilde]
        for _ in range(2, self.K + 1):
            cheb_polys.append(2 * L_tilde @ cheb_polys[-1] - cheb_polys[-2])
        self.cheb_polys = cheb_polys

        # GCN
        self.gcn = ChebGCN(self.F, self.gcn_out, self.K)

        # LSTM
        self.lstm = nn.LSTM(self.F + self.gcn_out, self.lstm_hidden, batch_first=True)

        # FC
        self.fc = nn.Linear(self.lstm_hidden, 1)

    def forward(self, X):
        # X: [B, T, N, F]
        X = X.to(self.device)
        B, T, N, F = X.shape
        H = []
        for t in range(T):
            X_t = X[:, t, :, :]  # [B, N, F]
            h_t = self.gcn(X_t, self.cheb_polys)  # [B, N, gcn_out]
            H.append(h_t)
        H = torch.stack(H, dim=1)  # [B, T, N, gcn_out]
        X_gcn = torch.cat([X, H], dim=-1)  # [B, T, N, F + gcn_out]
        target_seq = X_gcn[:, :, 0, :]  # [B, T, F + gcn_out] (target node 0)
        _, (h, _) = self.lstm(target_seq)
        out = self.fc(h.squeeze(0))
        return out

# Dataset for GC-LSTM
class GCLSTMDataset(Dataset):
    def __init__(self, pkl_file, config):
        with open(pkl_file, 'rb') as f:
            self.samples = pickle.load(f)
        self.config = config
        self.history_hours = config['history_hours']

        # Coords
        station_coords_dict = config["station_coords"]
        self.station_list = ['bjdst'] + sorted(k for k in station_coords_dict if k != 'bjdst')
        self.coords = torch.tensor([station_coords_dict[s] for s in self.station_list], dtype=torch.float32)  # [13, 2] (lon, lat)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        T = self.history_hours

        # Pollution: neighbors [12, T, 6], assume sample['pollution_seq'] [12, T, 6]
        pollution_neighbors = torch.tensor(sample['pollution_seq'][:, -T:, :], dtype=torch.float32)  # [12, T, 6], take last T hours
        pollution = torch.zeros(13, T, 6)
        pollution[1:, :, :] = pollution_neighbors
        pollution = pollution.permute(1, 0, 2)  # [T, 13, 6]

        # Weather: [T, 5], repeat for all nodes
        weather = torch.tensor(sample['weather_seq'][-T:, :], dtype=torch.float32)  # [T, 5]
        weather = weather.unsqueeze(1).repeat(1, 13, 1)  # [T, 13, 5]

        # Latlon: constant over T
        latlon = self.coords.unsqueeze(0).repeat(T, 1, 1)  # [T, 13, 2]

        # X
        X = torch.cat([pollution, weather, latlon], dim=-1)  # [T, 13, 13]

        # Target
        target = torch.tensor(sample['target'], dtype=torch.float32)

        return X, target

# Train function
def train(config):
    dataset = GCLSTMDataset(config["pkl_file"], config)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    device = config["device"]
    model = GCLSTM(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()

    early_stopper = EarlyStopping(patience=config["patience"])

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for X, target in train_loader:
            X, target = X.to(device), target.to(device).unsqueeze(1)  # target [B] -> [B,1]
            pred = model(X)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_train_loss = total_loss / len(train_dataset)

        model.eval()
        val_losses = []
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, target in val_loader:
                X, target = X.to(device), target.to(device).unsqueeze(1)
                pred = model(X)
                loss = loss_fn(pred, target)
                val_losses.append(loss.item() * X.size(0))
                y_true.extend(target.cpu().numpy().flatten())
                y_pred.extend(pred.cpu().numpy().flatten())
        avg_val_loss = sum(val_losses) / len(val_dataset)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test
    print("\nTesting...")
    model.eval()
    test_losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, target in test_loader:
            X, target = X.to(device), target.to(device).unsqueeze(1)
            pred = model(X)
            loss = loss_fn(pred, target)
            test_losses.append(loss.item() * X.size(0))
            y_true.extend(target.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())
    avg_test_loss = sum(test_losses) / len(test_dataset)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    print(f"Test Loss: {avg_test_loss:.4f}, R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# Config
BASE_CONFIG = {
    "history_hours": 24,  # Updated to match the dataset
    "batch_size": 16,
    "num_epochs": 150,
    "learning_rate": 1e-3,
    "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 20,
    "station_coords": {
        'bjdst': (116.300, 39.917),  # target
        'Wanshouxigong': (116.352, 39.878),
        'Dingling': (116.22, 40.292),
        'Dongsi': (116.417, 39.929),
        'Tiantan': (116.407, 39.886),
        'Nongzhanguan': (116.461, 39.937),
        'Guanyuan': (116.339, 39.929),
        'Haidingquwanliu': (116.287, 39.987),
        'Shunyixincheng': (116.655, 40.127),
        'Huairouzhen': (116.628, 40.328),
        'Changpingzhen': (116.23, 40.217),
        'Aotizhongxin': (116.397, 39.982),
        'Gucheng': (116.184, 39.914)
    }
}

if __name__ == "__main__":
    config = BASE_CONFIG
    train(config)