import os
import pickle
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
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
# 2. 数据集定义 (Dataset Definition)
# =============================================================================

class AirQualityDataset(Dataset):
    """
    自定义数据集类。
    注意：尽管DCCN-ALSTM模型只使用图像，但为了保持与您项目结构的兼容性，
    我们仍然加载并返回所有数据，模型在forward时会忽略不需要的部分。
    """
    def __init__(self, pkl_file, config):
        self.config = config
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 根据论文，图像进行了归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 图像序列
        history_hours = self.config.get("history_hours", 24)
        imgs = []
        # 只加载模型需要的历史长度
        image_paths_to_load = sample['images'][-history_hours:]
        for img_path in image_paths_to_load:
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    imgs.append(img)
                else:
                    # 如果图像不存在，用零张量填充
                    imgs.append(torch.zeros(3, 224, 224))
            except Exception as e:
                print(f"警告: 加载图像 {img_path} 时出错: {e}。将使用零张量代替。")
                imgs.append(torch.zeros(3, 224, 224))
        
        # 确保序列长度一致
        if len(imgs) < history_hours:
            num_padding = history_hours - len(imgs)
            imgs = [torch.zeros(3, 224, 224)] * num_padding + imgs

        imgs = torch.stack(imgs)

        # 污染物序列 (为保持兼容性而加载)
        pollution_seq = torch.tensor(sample['pollution_seq'], dtype=torch.float32)

        # 气象序列 (为保持兼容性而加载)
        weather_seq = torch.tensor(sample['weather_seq'], dtype=torch.float32)

        # 目标值
        target = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)

        # 邻接矩阵 (DCCN-ALSTM不使用，返回占位符)
        adj_mask, adj_phys = torch.ones(12), torch.ones(12)

        return imgs, pollution_seq, weather_seq, (adj_mask, adj_phys), target

# =============================================================================
# 3. DCCN-ALSTM 模型定义 (Model Definition)
# =============================================================================

class DCCN_ALSTM(nn.Module):
    """
    复现论文中的 DCCN-ALSTM 模型。
    DCCN: DenseNet for spatial feature extraction.
    ALSTM: Attention-based LSTM for temporal modeling.
    """
    def __init__(self, lstm_hidden_dim=128, num_lstm_layers=1, use_pretrained=True):
        super().__init__()
        
        # 1. DCCN (DenseNet-121) 图像特征提取器
        weights = models.DenseNet121_Weights.DEFAULT if use_pretrained else None
        densenet = models.densenet121(weights=weights)
        self.cnn_feature_dim = densenet.classifier.in_features  # 输出维度为 1024
        self.cnn_backbone = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. Attention-based LSTM (ALSTM) 时间序列建模器
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # 3. 全连接回归头
        # 论文中将上下文向量和最后隐藏状态拼接，所以维度是 2 * hidden_dim
        self.regressor = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, imgs, **kwargs):
        """
        模型前向传播。
        **kwargs 用于接收数据加载器传来的其他无用数据 (如 pollution, weather)。
        """
        # imgs 的形状: [B, T, C, H, W] (B: batch_size, T: 序列长度)
        B, T, C, H, W = imgs.shape
        
        # --- DCCN 特征提取 ---
        imgs_reshaped = imgs.view(B * T, C, H, W)
        cnn_out = self.cnn_backbone(imgs_reshaped)
        cnn_out = self.avgpool(cnn_out)
        cnn_out = torch.flatten(cnn_out, 1)
        feature_sequence = cnn_out.view(B, T, -1)
        
        # --- ALSTM 时间建模 ---
        lstm_outputs, (h_n, _) = self.lstm(feature_sequence)
        last_hidden_state = h_n[-1]
        
        # --- Attention Mechanism ---
        query = last_hidden_state.unsqueeze(1)
        keys = lstm_outputs
        attn_scores = torch.bmm(keys, query.transpose(1, 2)).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        # 拼接上下文向量和最后一个隐藏状态
        final_feature = torch.cat([context_vector, last_hidden_state], dim=1)
        
        # --- 回归预测 ---
        prediction = self.regressor(final_feature)
        
        return prediction

# =============================================================================
# 4. 训练与评估主函数 (Main Training & Evaluation Logic)
# =============================================================================

def train_and_evaluate(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # --- 数据准备 ---
    dataset = AirQualityDataset(config["pkl_file"], config)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    print(f"\n数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    # --- 模型、优化器、损失函数 ---
    device = config["device"]
    model = DCCN_ALSTM().to(device)
    
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

    # --- 训练与验证循环 ---
    print("\n[开始训练]")
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for imgs, _, _, _, target in train_loader:
            imgs, target = imgs.to(device), target.to(device)
            
            pred = model(imgs)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, _, _, _, target in val_loader:
                imgs, target = imgs.to(device), target.to(device)
                pred = model(imgs)
                loss = loss_fn(pred, target)
                total_val_loss += loss.item() * imgs.size(0)
                y_true.append(target.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
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
            save_path = os.path.join(config['save_dir'], "dccn_alstm_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> ✅ 模型已保存至 {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ 早停机制触发于 epoch {epoch+1}")
            break

    # --- 测试 ---
    print("\n[开始测试]")
    best_model_path = os.path.join(config['save_dir'], "dccn_alstm_best.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    total_test_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, _, _, _, target in test_loader:
            imgs, target = imgs.to(device), target.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, target)
            total_test_loss += loss.item() * imgs.size(0)
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            
    avg_test_loss = total_test_loss / len(test_dataset)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print("\n[测试结果]")
    print(f"  - Loss: {avg_test_loss:.4f}")
    print(f"  - R2  : {r2:.3f}")
    print(f"  - MAE : {mae:.3f}")
    print(f"  - RMSE: {rmse:.3f}")

    results = {"test_loss": avg_test_loss, "r2": r2, "mae": mae, "rmse": rmse}
    result_path = os.path.join(config['save_dir'], "dccn_alstm_test_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 5. 主程序入口 (Main Execution Block)
# =============================================================================

if __name__ == '__main__':
    # --- 实验配置 ---
    DCCN_ALSTM_CONFIG = {
        "name": "DCCN_ALSTM_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "./checkpoints/DCCN_ALSTM_baseline",
        
        # 模型与训练参数 (参考论文)
        "history_hours": 4,        # 论文中 K=4
        "learning_rate": 1e-3,     # 论文中为 0.001
        "batch_size": 32,         # 论文中为 128
        "num_epochs": 150,
        "patience": 15,            # 早停耐心值
    }

    # 创建保存目录
    os.makedirs(DCCN_ALSTM_CONFIG["save_dir"], exist_ok=True)
    
    # 启动训练与评估
    train_and_evaluate(DCCN_ALSTM_CONFIG)