"""
数据处理工具模块
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, List, Any


def safe_float(value: Any) -> float:
    """
    安全转换为浮点数

    Args:
        value: 输入值

    Returns:
        浮点数，转换失败返回0.0
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def normalize_timestamp(
    timestamp: Union[str, datetime, pd.Timestamp],
    format: str = None
) -> pd.Timestamp:
    """
    标准化时间戳

    Args:
        timestamp: 输入时间戳
        format: 时间格式（可选）

    Returns:
        标准化的时间戳
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp
    elif isinstance(timestamp, datetime):
        return pd.Timestamp(timestamp)
    elif isinstance(timestamp, str):
        if format:
            return pd.to_datetime(timestamp, format=format)
        else:
            return pd.to_datetime(timestamp)
    else:
        return pd.Timestamp.now()


def create_mask(
    data: np.ndarray,
    missing_value: float = np.nan
) -> np.ndarray:
    """
    创建数据掩码

    Args:
        data: 输入数据
        missing_value: 缺失值标记

    Returns:
        二值掩码数组（1=有效，0=缺失）
    """
    if np.isnan(missing_value):
        mask = (~np.isnan(data)).astype(float)
    else:
        mask = (data != missing_value).astype(float)
    return mask


def fill_missing_values(
    data: np.ndarray,
    method: str = "forward",
    fill_value: float = 0.0
) -> np.ndarray:
    """
    填充缺失值

    Args:
        data: 输入数据
        method: 填充方法（forward, backward, mean, zero）
        fill_value: 自定义填充值

    Returns:
        填充后的数据
    """
    result = np.asarray(data, dtype=float).copy()
    was_1d = result.ndim == 1
    if was_1d:
        result = result.reshape(-1, 1)

    valid = create_mask(result).astype(bool)

    if method == "forward":
        for col in range(result.shape[1]):
            for row in range(1, result.shape[0]):
                if not valid[row, col]:
                    result[row, col] = result[row - 1, col]
                    valid[row, col] = valid[row - 1, col]
    elif method == "backward":
        for col in range(result.shape[1]):
            for row in range(result.shape[0] - 2, -1, -1):
                if not valid[row, col]:
                    result[row, col] = result[row + 1, col]
                    valid[row, col] = valid[row + 1, col]
    elif method == "mean":
        for col in range(result.shape[1]):
            col_valid = valid[:, col]
            if col_valid.any():
                result[~col_valid, col] = result[col_valid, col].mean()
    elif method == "zero":
        result[~valid] = 0.0
    else:
        result[~valid] = fill_value

    return result.reshape(-1) if was_1d else result

    if method == "forward":
        # 前向填充
        mask = create_mask(result)
        for i in range(1, len(result)):
            if not mask[i]:
                result[i] = result[i-1]
    elif method == "backward":
        # 后向填充
        mask = create_mask(result)
        for i in range(len(result)-2, -1, -1):
            if not mask[i]:
                result[i] = result[i+1]
    elif method == "mean":
        # 均值填充
        mask = create_mask(result)
        if mask.sum() > 0:
            mean_value = result[mask.astype(bool)].mean()
            result[~mask.astype(bool)] = mean_value
    elif method == "zero":
        # 零值填充
        result[~create_mask(result).astype(bool)] = 0.0
    else:
        # 自定义值填充
        result[~create_mask(result).astype(bool)] = fill_value

    return result


def normalize_data(
    data: np.ndarray,
    method: str = "minmax",
    feature_range: tuple = (0, 1)
) -> tuple:
    """
    数据标准化

    Args:
        data: 输入数据
        method: 标准化方法（minmax, zscore）
        feature_range: MinMax范围

    Returns:
        (标准化后的数据, 参数字典)
    """
    if method == "minmax":
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # 避免除零

        normalized = (data - min_val) / range_val
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

        params = {'min': min_val, 'max': max_val, 'range': range_val}

    elif method == "zscore":
        mean_val = data.mean(axis=0)
        std_val = data.std(axis=0)
        std_val[std_val == 0] = 1  # 避免除零

        normalized = (data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val}

    else:
        raise ValueError(f"未知的标准化方法: {method}")

    return normalized, params


def denormalize_data(
    normalized_data: np.ndarray,
    params: dict,
    method: str = "minmax",
    feature_range: tuple = (0, 1)
) -> np.ndarray:
    """
    反标准化数据

    Args:
        normalized_data: 标准化后的数据
        params: 标准化参数
        method: 标准化方法
        feature_range: MinMax范围

    Returns:
        原始尺度的数据
    """
    if method == "minmax":
        # 反向MinMax标准化
        range_val = params['range']
        min_val = params['min']
        max_val = params['max']

        data = (normalized_data - feature_range[0]) / (feature_range[1] - feature_range[0])
        data = data * range_val + min_val

    elif method == "zscore":
        # 反向Z-score标准化
        mean_val = params['mean']
        std_val = params['std']

        data = normalized_data * std_val + mean_val

    else:
        raise ValueError(f"未知的标准化方法: {method}")

    return data


def extract_time_features(timestamps: pd.Series) -> np.ndarray:
    """
    提取时间特征

    Args:
        timestamps: 时间戳序列

    Returns:
        时间特征数组 [hour, dayofweek, day, month]
    """
    features = np.column_stack([
        timestamps.dt.hour / 23.0,      # 小时特征 [0,1]
        timestamps.dt.dayofweek / 6.0,  # 星期特征 [0,1]
        timestamps.dt.day / 31.0,       # 日期特征 [0,1]
        timestamps.dt.month / 11.0      # 月份特征 [0,1]
    ])

    return features


def create_sequence_pairs(
    data: np.ndarray,
    seq_len: int,
    pred_len: int
) -> List[tuple]:
    """
    创建序列对用于训练/推理

    Args:
        data: 输入数据
        seq_len: 输入序列长度
        pred_len: 预测序列长度

    Returns:
        序列对列表 [(input_seq, target_seq), ...]
    """
    pairs = []
    total_len = seq_len + pred_len

    for i in range(len(data) - total_len + 1):
        input_seq = data[i:i+seq_len]
        target_seq = data[i+seq_len:i+total_len]
        pairs.append((input_seq, target_seq))

    return pairs
