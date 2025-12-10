import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import AirQualityDataset
from distance import haversine_distance
import math


def idw_predict(values, distances, p=2):
    """
    IDW预测
    values: ndarray [N] 周边站点的PM2.5浓度
    distances: ndarray [N] 周边站点到目标点的距离
    p: 幂参数
    """
    weights = 1 / (np.power(distances, p) + 1e-6)  # 避免除0
    pred = np.sum(values * weights) / np.sum(weights)
    return pred


def evaluate_idw(dataset, distances, p=2):
    preds, targets = [], []
    for sample in dataset:
        station_values, target = sample  # (list[N], float)
        station_values = np.array(station_values)

        pred = idw_predict(station_values, distances, p)

        preds.append(pred)
        targets.append(target)
    datalength = math.floor(len(dataset)*0.15)

    targets = targets[-datalength:]
    preds = preds[-datalength:]
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = mse ** 0.5
    # rmse = mean_squared_error(targets, preds, squared=False)
    r2 = r2_score(targets, preds)

    return mae, rmse, r2


if __name__ == "__main__":
    dataset = AirQualityDataset("/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl")

    '''beijing'''
    target_lat, target_lon = 39.917, 116.300  # bjdst
    stations_lat = np.array([39.878, 40.292, 39.929, 39.886, 39.937, 39.929,
                             39.987, 40.127, 40.328, 40.217, 39.982, 39.914])
    stations_lon = np.array([116.352, 116.220, 116.417, 116.407, 116.461, 116.339,
                             116.287, 116.655, 116.628, 116.230, 116.397, 116.184])

    '''shanghai'''
    # target_lat, target_lon = 31.2397, 121.4998   # dfmz
    # stations_lat = np.array([
    #     31.1110,   # Shiwuchang
    #     31.2715,   # Hongkou
    #     31.1613,   # Shangshida
    #     31.2728,   # Yangpu
    #     31.1514,   # Qingpu
    #     31.2230,   # Jingan
    #     31.1869,   # PDchuansha
    #     31.2105,   # PDxinqu
    #     31.2012    # PDzhangjiang
    # ])
    # stations_lon = np.array([
    #     121.5670,  # Shiwuchang
    #     121.4800,  # Hongkou
    #     121.4208,  # Shangshida
    #     121.5306,  # Yangpu
    #     121.1139,  # Qingpu
    #     121.4456,  # Jingan
    #     121.6986,  # PDchuansha
    #     121.5508,  # PDxinqu
    #     121.5874   # PDzhangjiang
    # ])


    # 计算距离
    distances = haversine_distance(target_lat, target_lon, stations_lat, stations_lon)

    # 评估IDW
    mae, rmse, r2 = evaluate_idw(dataset, distances, p=2)
    print(f"IDW Baseline -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
