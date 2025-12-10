from dataset import AirQualityDataset
from distance import haversine_distance
import numpy as np



dataset = AirQualityDataset("/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl") 
# ([47.0, 36.0, 69.0, 35.79999923706055, 43.0, 48.0, 63.0, 67.0, 49.0, 77.0, 62.0, 47.0], 62)

target_lat, target_lon = 39.917, 116.300  # bjdst
stations_lat = np.array([
    39.878,   # Wanshouxigong
    40.292,   # Dingling
    39.929,   # Dongsi
    39.886,   # Tiantan
    39.937,   # Nongzhanguan
    39.929,   # Guanyuan
    39.987,   # Haidingquwanliu
    40.127,   # Shunyixincheng
    40.328,   # Huairouzhen
    40.217,   # Changpingzhen
    39.982,   # Aotizhongxin
    39.914    # Gucheng
])

stations_lon = np.array([
    116.352,  # Wanshouxigong
    116.220,  # Dingling
    116.417,  # Dongsi
    116.407,  # Tiantan
    116.461,  # Nongzhanguan
    116.339,  # Guanyuan
    116.287,  # Haidingquwanliu
    116.655,  # Shunyixincheng
    116.628,  # Huairouzhen
    116.230,  # Changpingzhen
    116.397,  # Aotizhongxin
    116.184   # Gucheng
])

distances = haversine_distance(target_lat, target_lon, stations_lat, stations_lon)
#目标点到周边站点的距离 (km): [ 6.20357527 42.24955576 10.06614857  9.75663426 13.90762703  3.58346584  7.8621282  38.19776239 53.53852227 33.88614506 10.98220408  9.89874835]