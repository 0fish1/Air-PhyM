import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点之间的大圆距离（单位：公里）
    lat1, lon1: 目标点纬度、经度 (标量)
    lat2, lon2: 周边站点纬度、经度 (数组)
    """
    R = 6371.0  # 地球半径，单位 km

    # 转换为弧度
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # [N] 数组
