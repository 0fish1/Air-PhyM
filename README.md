# Air-PhyM: A Physically Inspired Multimodal Fusion Framework with Contrastive Alignment for PM2.5 Estimation

Official code for the paper *"Air-PhyM: A Physically Inspired Multimodal Fusion Framework with Contrastive Alignment for PM2.5 Estimation"*.


## Overview

Air-PhyM is a novel physically inspired multimodal deep learning framework for air quality estimation. The model effectively integrates physical mechanisms with data-driven approaches, combining weather imagery, pollutant monitoring data, and meteorological information to achieve accurate air quality estimation.


## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
cd code/Beijing    # or code/Shanghai

python run_experiment.py --exp PM_SCL          # Full model
```

## Project Structure

```
code/
├── common/          # Core model modules
├── Beijing/         # Beijing config (12 neighbor stations)
└── Shanghai/        # Shanghai config (9 neighbor stations)
data/
├── bj/              # Beijing: samples_48h.pkl + raw CSVs
├── sh/              # Shanghai: samples_48h.pkl + raw CSVs
```

## Dataset Format

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

## License

MIT