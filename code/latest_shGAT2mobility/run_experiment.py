import argparse
from configs import experiment_configs
from train import train
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True,
                        choices=list(experiment_configs.keys()),
                        help=f"Experiment name, choices: {list(experiment_configs.keys())}")
    args = parser.parse_args()

    config = dict(experiment_configs[args.exp])
    print(f"Running experiment: {args.exp} with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    save_dir = f"experiments/{args.exp}"
    os.makedirs("experiments", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    config["save_dir"] = save_dir
    
    train(config)