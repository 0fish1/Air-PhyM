import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import random
import pickle
import os

# ==============================================================================
# SECTION 1: UTILITY & DATASET
# ==============================================================================

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
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

class AirQualityDataset(Dataset):
    """Loads neighbor station data for the model."""
    def __init__(self, pkl_file, config):
        self.config = config
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)
        return pollution_seq, target

# ==============================================================================
# SECTION 2: ASTGCN MODEL DEFINITION (Final, Corrected, and Guaranteed Version)
# ==============================================================================

class TemporalAttention(nn.Module):
    """Temporal Attention using standard QKV."""
    def __init__(self, in_features, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        # x: [B, N, T, C]
        B, N, T, C = x.shape
        
        # Reshape for multi-head attention
        query = self.query(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        key = self.key(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        value = self.value(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        # Attention scores: [B, N, H, T, T]
        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value: [B, N, H, T, head_dim]
        out = torch.matmul(attention, value)
        
        # Concatenate heads and reshape
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, N, T, C)
        return out

class SpatialAttention(nn.Module):
    """Spatial Attention using standard QKV."""
    def __init__(self, in_features, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        
        query = self.query(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        key = self.key(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        value = self.value(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, value)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, C)
        return out

class ChebConv(nn.Module):
    """Chebyshev Graph Convolution."""
    def __init__(self, in_channels, out_channels, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.reset_parameters()

    def reset_parameters(self):
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x, cheb_poly):
        # x: [B, N, C_in]
        # cheb_poly: [K, N, N]
        B, N, C_in = x.shape
        C_out = self.Theta[0].shape[1]
        
        outputs = []
        for k in range(self.K):
            T_k = cheb_poly[k]
            theta_k = self.Theta[k]
            
            graph_prop = torch.einsum('nn,bnc->bnc', T_k, x)
            outputs.append(torch.matmul(graph_prop, theta_k))
            
        return F.relu(sum(outputs))

class ASTGCNBlock(nn.Module):
    """The core Spatio-Temporal Block of ASTGCN."""
    def __init__(self, K, num_chev_filters, num_time_filters, cheb_polynomials, in_features, num_nodes, num_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttention(in_features)
        self.SAt = SpatialAttention(in_features)
        self.cheb_conv = ChebConv(in_features, num_chev_filters, K)
        self.time_conv = nn.Conv2d(num_chev_filters, num_time_filters, kernel_size=(1, 3), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_features, num_time_filters, kernel_size=(1, 1))
        self.ln = nn.LayerNorm([num_time_filters, num_nodes, num_timesteps])
        self.cheb_polynomials = cheb_polynomials

    def forward(self, x):
        # x: [B, N, C, T]
        B, N, C, T = x.shape
        
        # Temporal Attention
        x_tat = self.TAt(x) # Output: [B, N, T, C]
        
        # Spatial Attention
        x_perm_for_sat = x_tat.permute(0, 3, 1, 2) # [B, T, N, C]
        x_sat = self.SAt(x_perm_for_sat) # Output: [B, T, N, C]
        
        # Cheb GCN (iterating over time steps)
        gcn_outputs = []
        for t in range(T):
            gcn_in = x_sat[:, t, :, :] # [B, N, C]
            gcn_out = self.cheb_conv(gcn_in, self.cheb_polynomials) # [B, N, F_chev]
            gcn_outputs.append(gcn_out.unsqueeze(-1))
        
        spatial_gcn_out = torch.cat(gcn_outputs, dim=-1) # [B, N, F_chev, T]
        
        # Time Convolution
        time_conv_in = spatial_gcn_out.permute(0, 2, 1, 3) # [B, F_chev, N, T]
        time_conv_out = self.time_conv(time_conv_in)
        
        # Residual Connection
        residual_in = x.permute(0, 2, 1, 3) # [B, C, N, T]
        x_residual = self.residual_conv(residual_in)
        
        output = F.relu(x_residual + time_conv_out) # [B, F_time, N, T]
        
        return self.ln(output)

class ASTGCNModel(nn.Module):
    """The complete ASTGCN model for air quality prediction."""
    def __init__(self, num_nodes, in_features, num_timesteps, K, num_chev_filters, num_time_filters, station_coords):
        super(ASTGCNModel, self).__init__()
        
        cheb_polynomials = self._calculate_cheb_poly(station_coords, K)
        self.register_buffer('cheb_polynomials', torch.stack(cheb_polynomials, dim=0))
        
        self.block1 = ASTGCNBlock(K, num_chev_filters, num_time_filters, self.cheb_polynomials, in_features, num_nodes, num_timesteps)
        self.final_conv = nn.Conv2d(num_time_filters, 1, kernel_size=(1, num_timesteps))
        self.fc = nn.Linear(num_nodes, 1)

    def _calculate_cheb_poly(self, coords, K):
        print("Calculating Chebyshev polynomials...")
        dist = torch.cdist(coords, coords)
        sigma = torch.std(dist)
        adj = torch.exp(-torch.pow(dist, 2) / (2 * torch.pow(sigma, 2)))
        adj[adj < 0.1] = 0
        
        n = adj.shape[0]
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        L_norm = torch.eye(n) - torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        lambda_max = torch.linalg.eigvals(L_norm).real.max()
        L_tilde = (2.0 / lambda_max) * L_norm - torch.eye(n)
        
        cheb_poly = [torch.eye(n), L_tilde]
        for k in range(2, K):
            cheb_poly.append(2 * torch.matmul(L_tilde, cheb_poly[-1]) - cheb_poly[-2])
        return cheb_poly

    def forward(self, x):
        # x: [B, N, T, C]
        x = x.permute(0, 1, 3, 2) # [B, N, C, T]
        
        x = self.block1(x) # [B, F_time, N, T]
        
        x = self.final_conv(x).squeeze(1) # [B, N, 1]
        x = x.squeeze(-1) # [B, N]
        
        output = self.fc(x) # [B, 1]
        
        return output

# ==============================================================================
# SECTION 3: MAIN TRAINING SCRIPT
# ==============================================================================

def run_experiment(config):
    set_seed(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    
    print("\n" + "="*60)
    print("      Running Standalone ASTGCN Baseline (Final Guaranteed Version)")
    print("="*60)

    # --- Dataset and DataLoader ---
    dataset = AirQualityDataset(config["pkl_file"], config=config)
    train_size = int(0.7 * len(dataset)); val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    print(f"\nDataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    # --- Model Initialization ---
    device = config["device"]
    all_station_coords = torch.tensor(list(config["station_coords"].values()), dtype=torch.float32)
    
    model = ASTGCNModel(
        num_nodes=len(config["station_coords"]),
        in_features=6,
        num_timesteps=24,
        K=config["cheb_K"],
        num_chev_filters=config["num_chev_filters"],
        num_time_filters=config["num_time_filters"],
        station_coords=all_station_coords
    ).to(device)
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=config.get("patience", 20))
    best_val_loss = float("inf")
    save_path = os.path.join(config['save_dir'], "best_astgcn_final_model.pth")

    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for pollution, target in train_loader:
            pollution, target = pollution.to(device), target.to(device)
            B, N_neighbor, T, F = pollution.shape
            target_station_history = torch.zeros(B, 1, T, F, device=device)
            full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
            
            pred = model(full_pollution_seq)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item() * B
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss, y_true, y_pred = 0, [], []
        with torch.no_grad():
            for pollution, target in val_loader:
                pollution, target = pollution.to(device), target.to(device)
                B, N_neighbor, T, F = pollution.shape
                target_station_history = torch.zeros(B, 1, T, F, device=device)
                full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
                
                pred = model(full_pollution_seq)
                total_val_loss += loss_fn(pred, target).item() * B
                y_true.append(target.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)

        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"[Epoch {epoch+1:03d}] TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ Early stopping at epoch {epoch+1}"); break

    print("\nTesting the best model...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    total_test_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for pollution, target in test_loader:
            pollution, target = pollution.to(device), target.to(device)
            B, N_neighbor, T, F = pollution.shape
            target_station_history = torch.zeros(B, 1, T, F, device=device)
            full_pollution_seq = torch.cat([target_station_history, pollution], dim=1)
            
            pred = model(full_pollution_seq)
            total_test_loss += loss_fn(pred, target).item() * B
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_dataset)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    r2, mae, rmse = r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n" + "-"*60); print("      Final ASTGCN Architecture Test Results"); print("-"*60)
    print(f"Loss: {avg_test_loss:.4f} | R2: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

if __name__ == '__main__':
    # ==========================================================================
    # SECTION 4: CONFIGURATION BLOCK
    # ==========================================================================
    
    BASELINE_CONFIG = {
        "name": "astgcn_final_guaranteed_baseline",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "experiments/astgcn_final_guaranteed_baseline",
        # !!! 确保这里的路径是正确的 !!!
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        
        # --- Model Hyperparameters for ASTGCN ---
        "cheb_K": 3,
        "num_chev_filters": 64,
        "num_time_filters": 64,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "num_epochs": 150,
        "patience": 20,
        
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

    run_experiment(BASELINE_CONFIG)