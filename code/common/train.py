from pathlib import Path; import sys
COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path: sys.path.insert(0, str(COMMON_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import random
import pickle
import os
from losses import ContrastiveLossWithLabelThreshold
from utils import EarlyStopping
from itertools import chain


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config, DatasetClass, ModelClass):
    """Generic training loop.

    Args:
        config: experiment config dict
        DatasetClass: AirQualityDataset class (city-specific or common)
        ModelClass: AirQualityModel class (city-specific or common)
    """
    set_seed(config["seed"])

    print("\nTraining config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    dataset = DatasetClass(config["pkl_file"], config["distances"], config)

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    device = config["device"]
    model = ModelClass(config).to(device)

    print("\nModel config:")
    print(f"  Image branch: {'enabled' if config['use_image'] else 'disabled'}")
    print(f"  Pollution branch: {'enabled' if config['use_pollution'] else 'disabled'}")
    print(f"  Fusion type: {config['fusion_type']}")
    print(f"  Dynamic edge: {'enabled' if config['dynamic_edge'] else 'disabled'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    contrastive_criterion = None
    if config["use_contrastive"]:
        contrastive_criterion = ContrastiveLossWithLabelThreshold(
            img_dim=config["img_hidden_dim"],
            num_dim=config["pollution_hidden_dim"],
            threshold=config.get("contrastive_threshold", 1.0)
        ).to(device)
        optimizer = torch.optim.Adam(
            chain(model.parameters(), contrastive_criterion.parameters()),
            lr=config["learning_rate"]
        )

    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    early_stopper = EarlyStopping(patience=config.get("patience", 10), delta=1e-4)

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, pollution, weather, wind_info, target) in enumerate(train_loader):
            imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
            if wind_info is not None:
                wind_speed, wind_dir = wind_info
                wind_info = (wind_speed.to(device), wind_dir.to(device))

            if config["use_contrastive"]:
                pred, img_embed, num_embed = model(imgs, pollution, weather, wind_info=wind_info)
                loss_pred = loss_fn(pred, target)
                loss_contrast = contrastive_criterion(img_embed, num_embed, target)
                loss = loss_pred + loss_contrast
            else:
                pred = model(imgs, pollution, weather, wind_info=wind_info)
                loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_dataset)

        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for imgs, pollution, weather, wind_info, target in val_loader:
                imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
                if wind_info is not None:
                    wind_speed, wind_dir = wind_info
                    wind_info = (wind_speed.to(device), wind_dir.to(device))
                pred = model(imgs, pollution, weather, wind_info=wind_info)
                loss = loss_fn(pred, target)
                val_losses.append(loss.item() * imgs.size(0))
                y_true.append(target.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        avg_val_loss = np.sum(val_losses) / len(val_dataset)

        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5

        print(f"[Epoch {epoch+1}] TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} R2={r2:.3f} MAE={mae:.3f} RMSE={rmse:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{config['save_dir']}/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("\nTesting best model on test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_losses, y_true, y_pred = [], [], []
    sample_results = []

    with torch.no_grad():
        for i, (imgs, pollution, weather, wind_info, target) in enumerate(test_loader):
            imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
            if wind_info is not None:
                wind_speed, wind_dir = wind_info
                wind_info = (wind_speed.to(device), wind_dir.to(device))
            pred = model(imgs, pollution, weather, wind_info=wind_info)
            loss = loss_fn(pred, target)
            test_losses.append(loss.item() * imgs.size(0))
            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

            if i < 10:
                for j in range(min(5, len(target))):
                    sample_results.append({
                        "Actual": target[j].item(),
                        "Predicted": pred[j].item(),
                        "Difference": (target[j] - pred[j]).abs().item()
                    })

    avg_test_loss = np.sum(test_losses) / len(test_dataset)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    print(f"\nTest Results - Loss={avg_test_loss:.4f} R2={r2:.3f} MAE={mae:.3f} RMSE={rmse:.3f}")

    results = {
        "test_loss": avg_test_loss,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "sample_results": sample_results
    }
    result_path = f"{config['save_dir']}/test_results.pkl"
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Test results saved to {result_path}")