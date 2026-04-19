from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.configs import build_base_config, build_experiment_configs

PROJECT_ROOT = CODE_DIR.parents[0]
TARGET_STATION = "bjdst"
STATION_COORDS = {
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
}

BASE_CONFIG = build_base_config(
    project_root=PROJECT_ROOT,
    data_subdir="bj",
    pkl_file="/data/bj/samples_48h.pkl",
    image_root_candidates=[
        "data/bj_img",
        "data/images/bj",
        "data/images",
    ],
    target_station=TARGET_STATION,
    station_coords=STATION_COORDS,
    site_nums=12,
)

experiment_configs = build_experiment_configs(BASE_CONFIG, contrastive_threshold=1)