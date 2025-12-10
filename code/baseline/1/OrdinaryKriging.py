import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import AirQualityDataset
from pykrige.ok import OrdinaryKriging
import math


def kriging_predict(stations_lon, stations_lat, station_values, target_lon, target_lat, variogram_model="spherical"):
    """
    Ordinary Kriging 预测
    stations_lon, stations_lat: ndarray [N] 已知监测站点坐标
    station_values: ndarray [N] 已知监测站点数值 (PM2.5)
    target_lon, target_lat: float, 目标点经纬度
    variogram_model: 半方差函数模型 (spherical, exponential, gaussian)
    """
    OK = OrdinaryKriging(
        stations_lon,
        stations_lat,
        station_values,
        variogram_model=variogram_model,
        variogram_parameters={"sill": 1, "range": 1, "nugget": 0.1},
        verbose=False,
        enable_plotting=False,
    )
    z, ss = OK.execute("points", np.array([target_lon]), np.array([target_lat]))
    return z[0]  # 返回预测值


def evaluate_kriging(dataset, stations_lon, stations_lat, target_lon, target_lat, variogram_model="spherical"):
    preds, targets = [], []
    for sample in dataset:
        station_values, target = sample  # (list[N], float)
        station_values = np.array(station_values)

        pred = kriging_predict(stations_lon, stations_lat, station_values, target_lon, target_lat, variogram_model)
        preds.append(pred)
        targets.append(target)

    datalength = math.floor(len(dataset) * 0.15)
    targets = targets[-datalength:]
    preds = preds[-datalength:]

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = mse ** 0.5
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
    # target_lat, target_lon = 31.2397, 121.4998   # 东方明珠
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

    # 评估 Ordinary Kriging
    mae, rmse, r2 = evaluate_kriging(dataset, stations_lon, stations_lat, target_lon, target_lat, variogram_model="spherical")
    print(f"Ordinary Kriging -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
