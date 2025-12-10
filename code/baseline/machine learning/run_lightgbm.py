import os
import pickle
import random
import numpy as np
import lightgbm as lgb
import optuna

from sklearn.preprocessing import StandardScaler
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

def train_and_evaluate_lgbm(config):
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
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    
    print(f"数据集划分: 训练集={len(y_train)}, 验证集={len(y_val)}, 测试集={len(y_test)}")

    # --- 超参数搜索 (Optuna) ---
    if config["find_best_params"]:
        print("\n正在使用 Optuna 搜索最优超参数...")
        
        def objective(trial):
            params = {
                'objective': 'regression_l1', # MAE, 对异常值更鲁棒
                'metric': 'rmse',
                'n_estimators': 500,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'seed': config["seed"],
                'boosting_type': 'gbdt',
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=config["n_trials"])
        
        best_params = study.best_params
        print(f"搜索完成。最优参数为: {best_params}")
    else:
        print("\n使用预设的超参数...")
        best_params = config["best_params"]

    # --- 使用最优参数训练最终模型 ---
    print("\n正在训练最终模型...")
    final_params = best_params.copy()
    final_params.update({
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 2000, # 增加 estimators 数量，配合早停
        'n_jobs': -1,
        'seed': config["seed"],
        'boosting_type': 'gbdt',
    })
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=True)])
    
    print("模型训练完成。")

    # --- 测试 ---
    print("\n[开始在测试集上评估...]")
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n[LightGBM Baseline 测试结果]")
    print(f"  - R2  : {r2:.3f}")
    print(f"  - MAE : {mae:.3f}")
    print(f"  - RMSE: {rmse:.3f}")

    # --- 特征重要性分析 ---
    print("\n[特征重要性]")
    feature_names = config["feature_names"]
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    for i in range(10): # 打印前10个最重要的特征
        idx = sorted_indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]}")

    # --- 保存结果 ---
    os.makedirs(config["save_dir"], exist_ok=True)
    results = {"r2": r2, "mae": mae, "rmse": rmse, "best_params": best_params}
    result_path = os.path.join(config['save_dir'], "lgbm_test_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 4. 主程序入口
# =============================================================================

if __name__ == '__main__':
    # 为特征重要性分析定义特征名称
    POLLUTANT_NAMES = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    WEATHER_NAMES = ['Temp', 'Press', 'RH', 'WindDir', 'WindSpeed']
    
    SPATIAL_MEAN_NAMES = [f'sp_mean_{p}' for p in POLLUTANT_NAMES]
    SPATIAL_STD_NAMES = [f'sp_std_{p}' for p in POLLUTANT_NAMES]
    TEMPORAL_MEAN_NAMES = [f'tp_mean_{p}' for p in POLLUTANT_NAMES]
    TEMPORAL_TREND_NAMES = [f'tp_trend_{p}' for p in POLLUTANT_NAMES]
    
    FEATURE_NAMES = (WEATHER_NAMES + SPATIAL_MEAN_NAMES + SPATIAL_STD_NAMES + 
                     TEMPORAL_MEAN_NAMES + TEMPORAL_TREND_NAMES + ['target_hist_mean'])

    LGBM_CONFIG = {
        "name": "LightGBM_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl",
        "seed": 3407,
        "save_dir": "./checkpoints/LightGBM_baseline",
        
        "history_len": 24,
        "feature_names": FEATURE_NAMES,
        
        # 超参数搜索开关。第一次运行时设为 True
        "find_best_params": True, 
        "n_trials": 30, # Optuna 尝试的次数
        
        # 如果 find_best_params 为 False，将使用这里的参数
        "best_params": {
            'learning_rate': 0.05,
            'num_leaves': 150,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        } 
    }

    train_and_evaluate_lgbm(LGBM_CONFIG)