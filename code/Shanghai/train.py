from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.train import train as shared_train
from common.dataset import AirQualityDataset
from common.model import AirQualityModel


def train(config):
    return shared_train(config, AirQualityDataset, AirQualityModel)