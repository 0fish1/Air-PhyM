import argparse
import os


def main(experiment_configs, train_func):
    """Generic experiment entry point.

    Args:
        experiment_configs: dict of experiment_name -> config
        train_func: city-specific train function (calls common.train)
    """
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

    train_func(config)