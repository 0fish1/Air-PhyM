import torch
import numpy as np

# 预计算 distances
def compute_distances(station_coords):
    target_lon, target_lat = station_coords['bjdst']
    distances = []
    for name, (lon, lat) in station_coords.items():
        if name != 'bjdst':
            lon1, lat1, lon2, lat2 = map(np.radians, [target_lon, target_lat, lon, lat])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            distances.append(dist)
    return distances

BASE_CONFIG = {
    "use_image": True,
    "use_pollution": True,
    "use_weather": False,  # 禁用气象分支
    "fusion_type": "symmetrical_gated_attention",  # "concat" 或 "attention" 
    "history_hours": 24,
    "site_nums": 12,
    "dynamic_edge": True,
    "dynamic_use_time": True,
    "dynamic_use_wind": True,
    "cnn_backbone": "resnet18",
    "img_hidden_dim": 128,
    "pollution_hidden_dim": 128,
    "mlp_hidden_dim": 64,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "num_epochs": 150,
    "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl",
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "seed": 3407,
    "save_dir": "./checkpoints",
    "station_coords": {
        'bjdst': (116.300, 39.917),  # 目标站点
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
    },
    "distances": compute_distances({
        'bjdst': (116.300, 39.917),
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
    }),
    # 对比学习
    "use_contrastive": False,
    # "contrastive_lambda": 0.05,
    # "contrastive_threshold": 1.0,
    "proj_dim": 64,  # 对比学习投影维度
    "patch_dim": 128,
    # 早停
    "patience": 20,  # 早停耐心值
    "k_fold": 2,  # K折交叉验证次数
}

experiment_configs = {
    "PM_SCL": {
        **BASE_CONFIG,
        "name": "PM_SCL",
        "use_contrastive": True,
    },
    "image_only": {
        **BASE_CONFIG,
        "name": "image_only",
        "use_pollution": False
    },
    "pollution_only": {
        **BASE_CONFIG,
        "name": "pollution_only",
        "use_image": False
    },
    "attention_fusion": {
        **BASE_CONFIG,
        "name": "attention_fusion",
        "fusion_type": "attention"
    },
    "no_dynamic_edge": {
        **BASE_CONFIG,
        "name": "no_dynamic_edge",
        "dynamic_edge": False
    },
    "PM": {
        **BASE_CONFIG,
        "name": "PM",
    }
}