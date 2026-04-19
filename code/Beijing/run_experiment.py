from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.run_experiment import main
from configs import experiment_configs
from train import train


if __name__ == "__main__":
    main(experiment_configs, train)