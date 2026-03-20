import torch
import numpy as np

# 预计算 distances
def compute_distances(station_coords):
    target_lon, target_lat = station_coords['dfmz']
    distances = []
    for name, (lon, lat) in station_coords.items():
        if name != 'dfmz':
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
    "fusion_type": "cross_attention",  # "concat" 或 "attention" 或 "transformer"
    "history_hours": 24,
    "site_nums": 9,
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
    "pkl_file": "/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl",
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "save_dir": "./checkpoints",
    "station_coords": {
        'dfmz': (121.4998, 31.2397),
        'Shiwuchang': (121.567, 31.111),
        'Hongkou': (121.4800, 31.2715),
        'Shangshida': (121.4208, 31.1613),
        'Yangpu': (121.5306, 31.2728),
        'Qingpu': (121.1139, 31.1514),
        'Jingan': (121.4456, 31.2230),
        'PDchuansha': (121.6986, 31.1869),
        'PDxinqu': (121.5508, 31.2105),
        'PDzhangjiang': (121.5874, 31.2012),
        },
    "distances": compute_distances({
        'dfmz': (121.4998, 31.2397),
        'Shiwuchang': (121.567, 31.111),
        'Hongkou': (121.4800, 31.2715),
        'Shangshida': (121.4208, 31.1613),
        'Yangpu': (121.5306, 31.2728),
        'Qingpu': (121.1139, 31.1514),
        'Jingan': (121.4456, 31.2230),
        'PDchuansha': (121.6986, 31.1869),
        'PDxinqu': (121.5508, 31.2105),
        'PDzhangjiang': (121.5874, 31.2012)
    }),
    # 对比学习
    "use_contrastive": True,
    "contrastive_lambda": 0.05,
    "proj_dim": 64,  # 对比学习投影维度
    "patch_dim": 128,
    # 早停
    "patience": 20,  # 早停耐心值
    "k_fold": 2,  # K折交叉验证次数
}

experiment_configs = {
    "full_model": {
        **BASE_CONFIG,
        "name": "full_model"
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
    "no_image": {
        **BASE_CONFIG,
        "name": "no_image",
        "use_image": False
    },
    "no_pollution": {
        **BASE_CONFIG,
        "name": "no_pollution",
        "use_pollution": False
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
    }
}