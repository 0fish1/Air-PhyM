from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.configs import build_base_config, build_experiment_configs

PROJECT_ROOT = CODE_DIR.parents[0]
TARGET_STATION = "dfmz"
STATION_COORDS = {
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
}

BASE_CONFIG = build_base_config(
    project_root=PROJECT_ROOT,
    data_subdir="sh",
    pkl_file="/data/sh/samples_48h.pkl",
    image_root_candidates=[
        "data/sh_img",
        "data/images/sh",
        "data/images",
    ],
    target_station=TARGET_STATION,
    station_coords=STATION_COORDS,
    site_nums=9,
)

experiment_configs = build_experiment_configs(BASE_CONFIG, contrastive_threshold=3)