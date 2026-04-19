import torch
import numpy as np
from pathlib import Path


def compute_distances(station_coords):
    """Compute haversine distances from target to each neighbor station."""
    target_key = [k for k in station_coords if 'dst' in k.lower() or station_coords.get('_target') == k]
    if not target_key:
        # fallback: first entry is target
        target_key = list(station_coords.keys())[0]
    else:
        target_key = target_key[0]

    target_lon, target_lat = station_coords[target_key]
    distances = []
    for name, (lon, lat) in station_coords.items():
        if name != target_key and name != '_target':
            lon1, lat1, lon2, lat2 = map(np.radians, [target_lon, target_lat, lon, lat])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            distances.append(dist)
    return distances


def build_base_config(project_root, data_subdir, target_station, station_coords,
                      site_nums, pkl_file=None, image_root_candidates=None):
    """Build base config shared across all experiments.

    Args:
        project_root: Path to APM project root
        data_subdir: "bj" or "sh"
        target_station: key name for target station (e.g. "bjdst", "dfmz")
        station_coords: dict of station_name -> (lon, lat)
        site_nums: number of neighbor stations
        pkl_file: absolute path to samples pkl file (required)
        image_root_candidates: list of relative paths to try for image data
    """
    project_root = Path(project_root)
    if pkl_file is None:
        pkl_file = str(project_root / "data" / data_subdir / "samples_48h.pkl")

    # find image root
    image_root = None
    if image_root_candidates:
        for candidate in image_root_candidates:
            candidate_path = project_root / candidate
            if candidate_path.exists():
                image_root = str(candidate_path)
                break

    return {
        "use_image": True,
        "use_pollution": True,
        "fusion_type": "gated",
        "history_hours": 24,
        "site_nums": site_nums,
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
        "pkl_file": pkl_file,
        "image_root": image_root,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 3407,
        "save_dir": "./checkpoints",
        "station_coords": station_coords,
        "target_station_key": target_station,
        "distances": compute_distances(station_coords),
        "use_contrastive": False,
        "patience": 20,
    }


def build_experiment_configs(base_config, contrastive_threshold=1):
    """Build experiment variant configs from a base config."""
    return {
        "PM_SCL": {
            **base_config,
            "name": "PM_SCL",
            "use_contrastive": True,
        },
        "image_only": {
            **base_config,
            "name": "image_only",
            "use_pollution": False
        },
        "pollution_only": {
            **base_config,
            "name": "pollution_only",
            "use_image": False
        },
        "attention_fusion": {
            **base_config,
            "name": "attention_fusion",
            "fusion_type": "attention"
        },
        "no_dynamic_edge": {
            **base_config,
            "name": "no_dynamic_edge",
            "dynamic_edge": False
        },
        "PM": {
            **base_config,
            "name": "PM",
        }
    }