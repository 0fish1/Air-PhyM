import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import random
import pickle
import os

# ==============================================================================
# SECTION 1: UTILITY CLASSES AND FUNCTIONS
# (Dependencies that were previously in separate files)
# ==============================================================================

def set_seed(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class EarlyStopping:
    """Utility to stop training when a metric has stopped improving."""
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

# ==============================================================================
# SECTION 2: DATASET LOADER
# (Adapted from your dataset.py to be self-contained)
# ==============================================================================

class AirQualityDataset(Dataset):
    """
    Dataset class adapted for the baseline.
    It loads all data but the training loop will only use what's necessary.
    The 'mode' parameter ensures the pollution data has the correct shape for the baseline.
    """
    def __init__(self, pkl_file, config, mode="baseline"):
        self.config = config
        self.mode = mode
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

        # Images (loaded but not used by the baseline model)
        imgs = []
        for img_path in sample['images']:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                imgs.append(self.transform(img))
            else:
                imgs.append(torch.zeros(3, 224, 224))
        imgs = torch.stack(imgs) if imgs else torch.zeros(1, 3, 224, 224)

        # Pollution sequence
        # CRITICAL: For the baseline, we need ALL stations (target + neighbors)
        # The original code sliced this to only neighbors.

            # Keep all 13 stations for GC-LSTM processing
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)



        # Weather sequence (loaded but not used by baseline)
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

        # Target value
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)
        
        # Dummy adjacency info (not used by baseline, but keeps return signature consistent)
        adj_mask = torch.ones(self.config.get("site_nums", 12))

        return imgs, pollution_seq, weather_seq, adj_mask, target

# ==============================================================================
# SECTION 3: GC-LSTM MODEL DEFINITION
# ==============================================================================

class GCLSTMCell(nn.Module):
    """A single Graph Convolutional LSTM Cell."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.lstm_gates = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, adj, h_prev, c_prev):
        combined_input = torch.cat([x_t, h_prev], dim=-1)
        message = torch.relu(self.gcn_linear(combined_input))
        adj_batch = adj.unsqueeze(0).expand(x_t.shape[0], -1, -1)
        aggregated_message = torch.bmm(adj_batch, message)
        gates = self.lstm_gates(aggregated_message)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=-1)
        c_next = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class GCLSTMModel(nn.Module):
    """The full GC-LSTM model using a static, distance-based graph."""
    def __init__(self, num_stations, pollution_dim, hidden_dim, station_coords):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gclstm_cell = GCLSTMCell(input_dim=pollution_dim, hidden_dim=hidden_dim)
        self.register_buffer('adj', self._create_static_adj(station_coords))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _create_static_adj(self, coords):
        print("Creating static distance-based adjacency matrix for GC-LSTM baseline...")
        dist = torch.cdist(coords, coords)
        sigma = torch.std(dist)
        adj = torch.exp(-torch.pow(dist, 2) / (2 * torch.pow(sigma, 2)))
        row_sum = adj.sum(1)
        d_inv = torch.pow(row_sum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        norm_adj = torch.mm(torch.diag(d_inv), adj)
        print("Adjacency matrix created.")
        return norm_adj

    def forward(self, pollution_seq):
        B, N, T, _ = pollution_seq.shape
        device = pollution_seq.device
        h = torch.zeros(B, N, self.hidden_dim, device=device)
        c = torch.zeros(B, N, self.hidden_dim, device=device)
        for t in range(T):
            h, c = self.gclstm_cell(pollution_seq[:, :, t, :], self.adj, h, c)
        return self.regressor(h[:, 0, :])

# ==============================================================================
# SECTION 4: MAIN TRAINING AND EVALUATION SCRIPT
# ==============================================================================

def run_experiment(config):
    """Main function to run the GC-LSTM baseline experiment."""
    set_seed(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    
    print("\n" + "="*50)
    print("      Running Standalone GC-LSTM Baseline")
    print("="*50)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # --- Dataset and DataLoader Setup ---
    dataset = AirQualityDataset(config["pkl_file"], config=config, mode="baseline")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    print(f"\nDataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    # --- Model Initialization ---
    device = config["device"]
    all_station_coords = torch.tensor(list(config["station_coords"].values()), dtype=torch.float32)
    
    model = GCLSTMModel(
        num_stations=len(config["station_coords"]),
        pollution_dim=6,
        hidden_dim=config["gclstm_hidden_dim"],
        station_coords=all_station_coords
    ).to(device)
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=config.get("patience", 20))
    best_val_loss = float("inf")
    save_path = os.path.join(config['save_dir'], "best_baseline_model.pth")

    # --- Main Training Loop ---
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for _, pollution, _, _, target in train_loader:
            pollution, target = pollution.to(device), target.to(device)

            # ==================== 代码修复点 (核心) ====================
            # `pollution` 张量从数据加载器出来时是 [B, 12, T, F]
            # 我们需要将其变为 [B, 13, T, F] 以匹配模型的13个节点
            B, N, T, F = pollution.shape
            # 为目标站点创建一个全零的历史数据张量
            target_station_history = torch.zeros(B, 1, T, F, device=device)
            # 将目标站点(index 0)与12个邻居站点拼接起来
            full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
            
            # 使用拼接后的完整数据进行预测
            pred = model(full_pollution_seq)
            # =========================================================

            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * pollution.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        # --- Validation Loop ---
        model.eval()
        total_val_loss, y_true, y_pred = 0, [], []
        with torch.no_grad():
            for _, pollution, _, _, target in val_loader:
                pollution, target = pollution.to(device), target.to(device)
                
                # ==================== 同样应用修复 ====================
                B, N, T, F = pollution.shape
                target_station_history = torch.zeros(B, 1, T, F, device=device)
                full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
                pred = model(full_pollution_seq)
                # ====================================================

                total_val_loss += loss_fn(pred, target).item() * pollution.size(0)
                y_true.append(target.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)

        # --- Metrics and Logging ---
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"[Epoch {epoch+1:03d}] TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

        # --- Model Saving and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    # --- Final Testing ---
    print("\nTesting the best baseline model...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    total_test_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for _, pollution, _, _, target in test_loader:
            pollution, target = pollution.to(device), target.to(device)

            # ==================== 同样应用修复 ====================
            B, N, T, F = pollution.shape
            target_station_history = torch.zeros(B, 1, T, F, device=device)
            full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
            pred = model(full_pollution_seq)
            # ====================================================

            total_test_loss += loss_fn(pred, target).item() * pollution.size(0)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    avg_test_loss = total_test_loss / len(test_dataset)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n" + "-"*50)
    print("      Standalone GC-LSTM Baseline Test Results")
    print("-"*50)
    print(f"Loss: {avg_test_loss:.4f} | R2: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    
    results = {"test_loss": avg_test_loss, "r2": r2, "mae": mae, "rmse": rmse}
    result_path = os.path.join(config['save_dir'], "test_results.pkl")
    with open(result_path, 'wb') as f: pickle.dump(results, f)
    print(f"✅ Test results saved to {result_path}")


if __name__ == '__main__':
    # ==========================================================================
    # SECTION 5: CONFIGURATION BLOCK
    # (All settings are in one place for easy modification)
    # ==========================================================================
    
    BASELINE_CONFIG = {
        # --- Experiment Settings ---
        "name": "gc_lstm_standalone_baseline",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "experiments/gc_lstm_standalone_baseline",

        # --- Data Settings ---
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        
        # --- Model Hyperparameters ---
        "gclstm_hidden_dim": 128,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "num_epochs": 150,
        "patience": 20,
        
        # --- Station Coordinates (Order must be target first, then neighbors) ---
        "station_coords": {
            'bjdst': (39.917, 116.300),
            'Wanshouxigong': (39.878, 116.352), 'Dingling': (40.292, 116.22),
            'Dongsi': (39.929, 116.417), 'Tiantan': (39.886, 116.407),
            'Nongzhanguan': (39.937, 116.461), 'Guanyuan': (39.929, 116.339),
            'Haidingquwanliu': (39.987, 116.287), 'Shunyixincheng': (40.127, 116.655),
            'Huairouzhen': (40.328, 116.628), 'Changpingzhen': (40.217, 116.23),
            'Aotizhongxin': (39.982, 116.397), 'Gucheng': (39.914, 116.184)
        },
    }

    # Run the experiment with the defined configuration
    run_experiment(BASELINE_CONFIG)