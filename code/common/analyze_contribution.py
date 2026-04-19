from pathlib import Path; import sys
COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path: sys.path.insert(0, str(COMMON_DIR))

import torch
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def analyze(parsed_args, experiment_configs, DatasetClass, ModelClass):
    """Analyze multimodal contribution weights on the test split.

    Args:
        parsed_args: argparse Namespace with --exp and --num-samples
        experiment_configs: dict of experiment_name -> config
        DatasetClass: city-specific AirQualityDataset
        ModelClass: city-specific AirQualityModel
    """
    config = dict(experiment_configs[parsed_args.exp])
    device = config["device"]

    dataset = DatasetClass(config["pkl_file"], config["distances"], config)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_indices = range(train_size + val_size, total)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = ModelClass(config).to(device)
    save_path = f"{config['save_dir']}/best_model.pth"
    model.load_state_dict(torch.load(save_path))
    model.eval()

    alphas = []
    num_samples = parsed_args.num_samples

    with torch.no_grad():
        for i, (imgs, pollution, weather, wind_info, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            imgs = imgs.to(device)
            pollution = pollution.to(device)
            weather = weather.to(device)
            if wind_info is not None:
                wind_speed, wind_dir = wind_info
                wind_info = (wind_speed.to(device), wind_dir.to(device))

            pred, alpha = model(imgs, pollution, weather, wind_info=wind_info, return_contribution=True)
            alphas.append(alpha.cpu().numpy())
            print(f"Sample {i+1}: pred={pred.mean().item():.2f}, alpha={alpha.mean().item():.3f}")

    alphas = np.concatenate(alphas)
    print(f"\nContribution summary (alpha = image weight):")
    print(f"  Mean alpha: {alphas.mean():.3f}")
    print(f"  Std alpha:  {alphas.std():.3f}")
    print(f"  Image contribution: {alphas.mean() * 100:.1f}%")
    print(f"  Pollution contribution: {(1 - alphas.mean()) * 100:.1f}%")