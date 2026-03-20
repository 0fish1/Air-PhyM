import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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

def train(config):
    set_seed(config["seed"])
    
    print("\n实际训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    from dataset import AirQualityDataset
    dataset = AirQualityDataset(config["pkl_file"], config["distances"], config)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    from model import AirQualityModel
    device = config["device"]
    model = AirQualityModel(config).to(device)
    
    print("\n模型结构差异:")
    print(f"图像分支: {'启用' if config['use_image'] else '禁用'}")
    print(f"污染物分支: {'启用' if config['use_pollution'] else '禁用'}")
    print(f"融合方式: {config['fusion_type']}")
    print(f"动态建边: {'启用' if config['dynamic_edge'] else '禁用'}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    print("\n模型组件验证:")
    print(f"CNN存在: {hasattr(model, 'cnn')}")
    print(f"污染物GRU存在: {hasattr(model, 'pollution_encoder')}")


    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")


    
    early_stopper = EarlyStopping(patience=config.get("patience", 10), delta=1e-4)

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0

        
        for batch_idx, (imgs, pollution, weather, adj, target) in enumerate(train_loader):
            imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
            adj_hybrid, adj_phys = adj
            adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
            
            if config["use_contrastive"]:
                contrastive_criterion = ContrastiveLossWithLabelThreshold(
                    img_dim=config["img_hidden_dim"],
                    num_dim=config["pollution_hidden_dim"],
                    threshold=1.0
                ).to(device)

                pred, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)
                loss_pred = loss_fn(pred, target)
                loss_contrast = contrastive_criterion(img_embed, num_embed, target)
                loss = loss_pred + loss_contrast
            else:
                pred = model(imgs, pollution, weather, adj_mask=adj_mask)
                loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_dataset)

        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for imgs, pollution, weather, adj, target in val_loader:
                imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
                adj_hybrid, adj_phys = adj
                adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
                pred = model(imgs, pollution, weather, adj_mask=adj_mask)
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
            print(f"✅ Best model saved to {save_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    print("\nTesting the best model on test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_losses, y_true, y_pred = [], [], []
    sample_results = []

    with torch.no_grad():
        for i, (imgs, pollution, weather, adj, target) in enumerate(test_loader):
            imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
            adj_hybrid, adj_phys = adj
            adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
            pred = model(imgs, pollution, weather, adj_mask=adj_mask)
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
    print(f"✅ Test results saved to {result_path}")









# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import numpy as np
# import random
# import pickle
# import os
# from losses import ContrastiveLossWithLabelThreshold
# from utils import EarlyStopping
# from itertools import chain
# import logging

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def train(config):
#     set_seed(config["seed"])
    
#     # Set up logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)
    
#     logger.info("\n实际训练配置:")
#     for k, v in config.items():
#         logger.info(f"  {k}: {v}")

#     from dataset import AirQualityDataset
#     dataset = AirQualityDataset(config["pkl_file"], config["distances"], config)

#     train_size = int(0.7 * len(dataset))
#     val_size = int(0.15 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, [train_size, val_size, test_size]
#     )

#     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

#     from model import AirQualityModel
#     device = config["device"]
#     model = AirQualityModel(config).to(device)
    
#     logger.info("\n模型结构差异:")
#     logger.info(f"图像分支: {'启用' if config['use_image'] else '禁用'}")
#     logger.info(f"污染物分支: {'启用' if config['use_pollution'] else '禁用'}")
#     logger.info(f"融合方式: {config['fusion_type']}")
#     logger.info(f"动态建边: {'启用' if config['dynamic_edge'] else '禁用'}")
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"\n总参数数量: {total_params:,}")
#     logger.info(f"可训练参数数量: {trainable_params:,}")
    
#     logger.info("\n模型组件验证:")
#     logger.info(f"CNN存在: {hasattr(model, 'cnn')}")
#     logger.info(f"污染物GRU存在: {hasattr(model, 'pollution_encoder')}")

#     contrastive_criterion = None
#     if config["use_contrastive"]:
#         contrastive_threshold = config.get("contrastive_threshold", 2)
#         contrastive_criterion = ContrastiveLossWithLabelThreshold(
#             img_dim=config["img_hidden_dim"],
#             num_dim=config["pollution_hidden_dim"],
#             threshold=contrastive_threshold
#         ).to(device)

#     if config["use_contrastive"]:
#         params_to_optimize = chain(model.parameters(), contrastive_criterion.parameters())
#         optimizer = torch.optim.Adam(params_to_optimize, lr=config["learning_rate"])
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

#     from torch.optim.lr_scheduler import ReduceLROnPlateau
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#     loss_fn = nn.MSELoss()
#     best_val_loss = float("inf")
    
#     early_stopper = EarlyStopping(patience=config.get("patience", 10), delta=1e-4)

#     for epoch in range(config["num_epochs"]):
#         model.train()
#         total_loss = 0
        
#         for batch_idx, (imgs, pollution, weather, adj, target) in enumerate(train_loader):
#             imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
#             adj_hybrid, adj_phys = adj
#             adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
            
#             if config["use_contrastive"]:
#                 pred, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)
#                 loss_pred = loss_fn(pred, target)
#                 loss_contrast = contrastive_criterion(img_embed, num_embed, target)
#                 loss = loss_pred + loss_contrast
#             else:
#                 pred = model(imgs, pollution, weather, adj_mask=adj_mask)
#                 loss = loss_fn(pred, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * imgs.size(0)
#         avg_train_loss = total_loss / len(train_dataset)

#         model.eval()
#         val_losses, y_true, y_pred = [], [], []
#         with torch.no_grad():
#             for imgs, pollution, weather, adj, target in val_loader:
#                 imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
#                 adj_hybrid, adj_phys = adj
#                 adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
#                 pred = model(imgs, pollution, weather, adj_mask=adj_mask)
#                 loss = loss_fn(pred, target)
#                 val_losses.append(loss.item() * imgs.size(0))
#                 y_true.append(target.cpu().numpy())
#                 y_pred.append(pred.cpu().numpy())
#         avg_val_loss = np.sum(val_losses) / len(val_dataset)

#         y_true_flat, y_pred_flat = np.concatenate(y_true), np.concatenate(y_pred)
#         if np.all(y_true_flat == y_true_flat[0]):  # Handle constant y_true
#             r2 = 0.0  # or np.nan, but set to 0 for safety
#         else:
#             r2 = r2_score(y_true_flat, y_pred_flat)
#         mae = mean_absolute_error(y_true_flat, y_pred_flat)
#         mse = mean_squared_error(y_true_flat, y_pred_flat)
#         rmse = mse ** 0.5

#         logger.info(f"[Epoch {epoch+1}] TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} R2={r2:.3f} MAE={mae:.3f} RMSE={rmse:.3f}")
        
#         scheduler.step(avg_val_loss)
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             save_path = f"{config['save_dir']}/best_model.pth"
#             try:
#                 torch.save(model.state_dict(), save_path)
#                 logger.info(f"✅ Best model saved to {save_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save model: {e}")

#         early_stopper(avg_val_loss)
#         if early_stopper.early_stop:
#             logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
#             break

#     logger.info("\nTesting the best model on test set...")
#     try:
#         model.load_state_dict(torch.load(save_path))
#     except Exception as e:
#         logger.error(f"Failed to load model: {e}")
#         return

#     model.eval()
#     test_losses, y_true, y_pred = [], [], []
#     sample_results = []

#     with torch.no_grad():
#         for i, (imgs, pollution, weather, adj, target) in enumerate(test_loader):
#             imgs, pollution, weather, target = imgs.to(device), pollution.to(device), weather.to(device), target.to(device)
#             adj_hybrid, adj_phys = adj
#             adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
#             pred = model(imgs, pollution, weather, adj_mask=adj_mask)
#             loss = loss_fn(pred, target)
#             test_losses.append(loss.item() * imgs.size(0))
#             y_true.append(target.cpu().numpy())
#             y_pred.append(pred.cpu().numpy())
            
#             if i < 10:
#                 for j in range(min(5, len(target))):
#                     sample_results.append({
#                         "Actual": target[j].item(),
#                         "Predicted": pred[j].item(),
#                         "Difference": (target[j] - pred[j]).abs().item()
#                     })

#     avg_test_loss = np.sum(test_losses) / len(test_dataset)
#     y_true_flat, y_pred_flat = np.concatenate(y_true), np.concatenate(y_pred)
#     if np.all(y_true_flat == y_true_flat[0]):
#         r2 = 0.0
#     else:
#         r2 = r2_score(y_true_flat, y_pred_flat)
#     mae = mean_absolute_error(y_true_flat, y_pred_flat)
#     mse = mean_squared_error(y_true_flat, y_pred_flat)
#     rmse = mse ** 0.5

#     logger.info(f"\nTest Results - Loss={avg_test_loss:.4f} R2={r2:.3f} MAE={mae:.3f} RMSE={rmse:.3f}")

#     results = {
#         "test_loss": avg_test_loss,
#         "r2": r2,
#         "mae": mae,
#         "rmse": rmse,
#         "sample_results": sample_results
#     }
#     result_path = f"{config['save_dir']}/test_results.pkl"
#     try:
#         with open(result_path, 'wb') as f:
#             pickle.dump(results, f)
#         logger.info(f"✅ Test results saved to {result_path}")
#     except Exception as e:
#         logger.error(f"Failed to save results: {e}")