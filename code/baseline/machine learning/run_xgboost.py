import os
import pickle
import random
import numpy as np
import xgboost as xgb

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# 1. 工具类与函数 (Utilities)
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# =============================================================================
# 2. 数据集定义与特征工程
# =============================================================================

class FeatureDataset:
    """
    为经典机器学习模型准备数据和进行特征工程。
    """
    def __init__(self, pkl_file, config):
        self.config = config
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")

    def process(self):
        X_features, y_targets = [], []
        history_len = self.config["history_len"]

        for sample in self.samples:
            neighbor_pollution_seq = sample['pollution_seq']
            weather_seq = sample['weather_seq']
            
            current_weather = weather_seq[-1, :]
            
            current_neighbor_pollution = neighbor_pollution_seq[:, -1, :]
            spatial_mean_features = np.mean(current_neighbor_pollution, axis=0)
            spatial_std_features = np.std(current_neighbor_pollution, axis=0)
            
            closest_neighbor_history = neighbor_pollution_seq[0, -history_len:, :]
            temporal_mean_features = np.mean(closest_neighbor_history, axis=0)
            temporal_trend_features = closest_neighbor_history[-1, :] - closest_neighbor_history[0, :]
            
            target_station_approx_history = closest_neighbor_history[:, 0]
            target_history_mean = np.mean(target_station_approx_history)

            feature_vector = np.concatenate([
                current_weather, spatial_mean_features, spatial_std_features,
                temporal_mean_features, temporal_trend_features, [target_history_mean]
            ])
            
            X_features.append(feature_vector)
            y_targets.append(sample['target'])

        return np.array(X_features), np.array(y_targets).flatten()

# =============================================================================
# 3. 训练与评估主函数
# =============================================================================

def train_and_evaluate_xgb(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # --- 数据准备与特征工程 ---
    dataset = FeatureDataset(config["pkl_file"], config)
    X_all, y_all = dataset.process()
    
    print(f"\n特征工程完成。特征矩阵 X 的形状: {X_all.shape}")

    # --- 数据集划分 ---
    num_samples = len(X_all)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices] # 验证集在这里没有用到，但保留划分以保持一致
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    
    print(f"数据集划分: 训练集={len(y_train)}, 验证集={len(y_val)}, 测试集={len(y_test)}")

    # --- 使用预设参数训练模型 ---
    print("\n正在使用预设参数训练模型 (无早停)...")
    
    model = xgb.XGBRegressor(**config["params"])
    
    # 直接在完整的训练集上训练固定的轮数
    model.fit(X_train, y_train, verbose=True)
    
    print("模型训练完成。")

    # --- 测试 ---
    print("\n[开始在测试集上评估...]")
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n[XGBoost (Legacy) Baseline 测试结果]")
    print(f"  - R2  : {r2:.3f}")
    print(f"  - MAE : {mae:.3f}")
    print(f"  - RMSE: {rmse:.3f}")

    # --- 特征重要性分析 ---
    print("\n[特征重要性]")
    feature_names = config["feature_names"]
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    for i in range(10):
        idx = sorted_indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # --- 保存结果 ---
    os.makedirs(config["save_dir"], exist_ok=True)
    results = {"r2": r2, "mae": mae, "rmse": rmse, "params": config["params"]}
    result_path = os.path.join(config['save_dir'], "xgb_legacy_test_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 4. 主程序入口
# =============================================================================

if __name__ == '__main__':
    POLLUTANT_NAMES = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    WEATHER_NAMES = ['Temp', 'Press', 'RH', 'WindDir', 'WindSpeed']
    
    SPATIAL_MEAN_NAMES = [f'sp_mean_{p}' for p in POLLUTANT_NAMES]
    SPATIAL_STD_NAMES = [f'sp_std_{p}' for p in POLLUTANT_NAMES]
    TEMPORAL_MEAN_NAMES = [f'tp_mean_{p}' for p in POLLUTANT_NAMES]
    TEMPORAL_TREND_NAMES = [f'tp_trend_{p}' for p in POLLUTANT_NAMES]
    
    FEATURE_NAMES = (WEATHER_NAMES + SPATIAL_MEAN_NAMES + SPATIAL_STD_NAMES + 
                     TEMPORAL_MEAN_NAMES + TEMPORAL_TREND_NAMES + ['target_hist_mean'])

    XGB_CONFIG = {
        "name": "XGBoost_legacy_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
        "seed": 3407,
        "save_dir": "./checkpoints/XGBoost_legacy_baseline",
        
        "history_len": 24,
        "feature_names": FEATURE_NAMES,
        
        # 预设的一组通用参数
        "params": {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 500, # 训练固定的500轮
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'n_jobs': -1,
            'seed': 3407,
        } 
    }

    train_and_evaluate_xgb(XGB_CONFIG)