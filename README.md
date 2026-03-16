# **Air-PhyM: A Physics-Informed Multimodal Fusion Framework with Contrastive Alignment for PM2.5 Estimation**  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### Overview

Air-PhyM is a novel physics-informed multimodal deep learning framework for air quality estimation. The model effectively integrates physical mechanisms with data-driven approaches, combining weather imagery, pollutant monitoring data, and meteorological information to achieve accurate air quality estimation.

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Air-PhyM.git
cd Air-PhyM

# Create conda environment
conda create -n airphym python=3.9
conda activate airphym

# Install dependencies
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install einops mmengine
pip install pillow matplotlib
pip install optuna lightgbm xgboost
pip install fastdtw scipy
```

### Quick Start

```bash
# Run the full model
cd code/latest_shGAT2mobility
python run_experiment.py --exp full_model

# Run ablation studies
python run_experiment.py --exp image_only
python run_experiment.py --exp pollution_only
python run_experiment.py --exp no_dynamic_edge
```

### Configuration

Key parameters in `configs.py`:

| Parameter         | Description                   | Default            |
| ----------------- | ----------------------------- | ------------------ |
| `use_image`       | Enable image branch           | True               |
| `use_pollution`   | Enable pollution branch       | True               |
| `fusion_type`     | Fusion method                 | "cross\_attention" |
| `dynamic_edge`    | Use dynamic edge construction | True               |
| `history_hours`   | Historical time window        | 24                 |
| `use_contrastive` | Enable contrastive learning   | True               |
| `batch_size`      | Training batch size           | 16                 |
| `num_epochs`      | Number of epochs              | 150                |
| `learning_rate`   | Learning rate                 | 1e-3               |

### Dataset Format

The dataset should be stored in pickle format (`samples_48h.pkl`) with the following structure:

```python
{
    'pollution_seq': np.array,  # Shape: [num_neighbors, time_steps, 6]
                                # Features: PM2.5, PM10, SO2, NO2, CO, O3
    'weather_seq': np.array,    # Shape: [time_steps, 5]
                                # Features: temperature, wind_dir, wind_speed, humidity, pressure
    'images': list,             # List of image file paths
    'target': float,            # Target PM2.5 value
}
```

<br />

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
