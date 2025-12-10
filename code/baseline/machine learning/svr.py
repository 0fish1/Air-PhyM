import os
import pickle
import random
import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

class SVRDataset:
    """
    为SVR模型准备数据和进行特征工程。
    """
    def __init__(self, pkl_file, config):
        self.config = config
        print(f"正在从 {pkl_file} 加载数据...")
        with open(pkl_file, "rb") as f:
            self.samples = pickle.load(f)
        print("数据加载完成。")

    def process(self):
        """
        处理所有样本，生成特征矩阵 X 和目标向量 y。
        """
        X_features = []
        y_targets = []
        
        history_len = self.config["history_len"]

        for sample in self.samples:
            # --- 1. 提取基础数据 ---
            # 邻居站污染序列: [N_neighbors, T, F_pollutants]
            neighbor_pollution_seq = sample['pollution_seq']
            # 全局气象序列: [T, F_weather]
            weather_seq = sample['weather_seq']
            
            # --- 2. 构建特征向量 ---
            # a) 当前时刻的气象特征 (5维)
            current_weather = weather_seq[-1, :]
            
            # b) 空间聚合特征 (邻居站在当前时刻的统计值)
            # 6个污染物 * 2个统计量 (mean, std) = 12维
            current_neighbor_pollution = neighbor_pollution_seq[:, -1, :]
            spatial_mean_features = np.mean(current_neighbor_pollution, axis=0)
            spatial_std_features = np.std(current_neighbor_pollution, axis=0)
            
            # c) 时间聚合特征 (最近邻居的历史统计值)
            # 我们假设第一个邻居站(Wanshouxigong)是最近的
            # 6个污染物 * 2个统计量 (mean, trend) = 12维
            closest_neighbor_history = neighbor_pollution_seq[0, -history_len:, :]
            temporal_mean_features = np.mean(closest_neighbor_history, axis=0)
            temporal_trend_features = closest_neighbor_history[-1, :] - closest_neighbor_history[0, :]
            
            # d) 拼接所有特征 (5 + 12 + 12 + 1 = 30维)
            # 最后一个特征是目标站自己的历史均值，作为强特征
            target_station_approx_history = closest_neighbor_history[:, 0] # PM2.5 history
            target_history_mean = np.mean(target_station_approx_history)

            feature_vector = np.concatenate([
                current_weather,
                spatial_mean_features,
                spatial_std_features,
                temporal_mean_features,
                temporal_trend_features,
                [target_history_mean]
            ])
            
            X_features.append(feature_vector)
            y_targets.append(sample['target'])

        return np.array(X_features), np.array(y_targets).flatten()

# =============================================================================
# 3. 训练与评估主函数
# =============================================================================

def train_and_evaluate_svr(config):
    set_seed(config["seed"])
    
    print("\n[配置信息]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # --- 数据准备与特征工程 ---
    dataset = SVRDataset(config["pkl_file"], config)
    X_all, y_all = dataset.process()
    
    print(f"\n特征工程完成。特征矩阵 X 的形状: {X_all.shape}")

    # --- 数据集划分 ---
    num_samples = len(X_all)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    
    print(f"数据集划分: 训练集={len(y_train)}, 验证集={len(y_val)}, 测试集={len(y_test)}")

    # --- 数据标准化 ---
    print("\n正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("标准化完成。")

    # --- 超参数搜索 (GridSearchCV) ---
    if config["find_best_params"]:
        print("\n正在使用 GridSearchCV 搜索最优超参数...")
        # param_grid = {
        #     'C': [0.1, 1, 10, 100],
        #     'gamma': ['scale', 'auto', 0.1, 1],
        #     'kernel': ['rbf']
        # }
        # param_grid = {
        #     'C': [1, 10, 100, 1000],
        #     'gamma': [0.001, 0.01, 0.1, 'scale'],
        #     'epsilon': [0.01, 0.1, 0.5],
        #     'kernel': ['rbf']
        # }

        param_grid = {
            'C': [1, 10],
            'gamma': [0.001, 0.01, 0.1, 'scale'],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf']
        }
        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(np.concatenate([X_train_scaled, X_val_scaled]), 
                        np.concatenate([y_train, y_val]))
        
        best_params = grid_search.best_params_
        print(f"搜索完成。最优参数为: {best_params}")
        model = grid_search.best_estimator_
    else:
        print("\n使用预设的超参数进行训练...")
        best_params = config["best_params"]
        model = SVR(**best_params)
        model.fit(X_train_scaled, y_train)
    
    print("模型训练完成。")

    # --- 测试 ---
    print("\n[开始在测试集上评估...]")
    predictions = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n[SVR Baseline 测试结果]")
    print(f"  - R2  : {r2:.3f}")
    print(f"  - MAE : {mae:.3f}")
    print(f"  - RMSE: {rmse:.3f}")

    # --- 保存结果 ---
    os.makedirs(config["save_dir"], exist_ok=True)
    results = {"r2": r2, "mae": mae, "rmse": rmse, "best_params": best_params}
    result_path = os.path.join(config['save_dir'], "svr_test_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 测试结果已保存至 {result_path}")

# =============================================================================
# 4. 主程序入口
# =============================================================================

if __name__ == '__main__':
    SVR_CONFIG = {
        "name": "SVR_baseline",
        "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl",
        "seed": 3407,
        "save_dir": "./checkpoints/SVR_baseline",
        
        "history_len": 24, # 用于计算时间聚合特征的历史长度
        
        # 超参数搜索开关。第一次运行时设为 True，后续可以设为 False 并填入找到的最优参数
        "find_best_params": True, 
        # 如果 find_best_params 为 False，将使用这里的参数
        "best_params": {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} 
    }

    train_and_evaluate_svr(SVR_CONFIG)