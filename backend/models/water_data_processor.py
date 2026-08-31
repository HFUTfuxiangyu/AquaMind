"""
水务数据处理器模块
"""
import pandas as pd
import numpy as np
import torch
import io
from typing import Dict, List, Tuple, Optional, Any
from config import settings
from utils.logger import get_logger
from utils.data_utils import (
    normalize_timestamp,
    create_mask,
    fill_missing_values,
    extract_time_features,
    safe_float
)

logger = get_logger(__name__)


class WaterDataProcessor:
    """水务数据专用的预处理器"""

    def __init__(self):
        """初始化数据处理器"""
        self.column_mapping = settings.column_mapping
        self.feature_columns = settings.feature_columns

    def csv_to_tensor(
        self,
        csv_content: str,
        max_length: int = None,
        normalization_override: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        将CSV数据转换为模型输入张量

        Args:
            csv_content: CSV格式字符串
            max_length: 最大序列长度

        Returns:
            包含张量的字典
        """
        try:
            # 解析CSV
            df = pd.read_csv(io.StringIO(csv_content))

            # 列名映射
            df = self._map_columns(df)

            # 处理时间戳
            df['timestamp'] = df['timestamp'].apply(normalize_timestamp)
            df = df.sort_values('timestamp')

            # 提取特征
            features = self._extract_features(df, normalization_override)

            # 限制序列长度
            if max_length is None:
                max_length = settings.max_sequence_length

            if len(features['values']) > max_length:
                # 取最后max_length个数据点
                for key in features:
                    if isinstance(features[key], np.ndarray):
                        features[key] = features[key][-max_length:]

            seq_count = len(features['values'])
            if seq_count:
                features['time_features'][:, 0] = np.arange(seq_count, dtype=float) / max(seq_count, 1)

            # 创建张量
            tensor_data = {
                'x': torch.tensor(features['values'], dtype=torch.float32),
                'x_mark': torch.tensor(features['time_features'], dtype=torch.float32),
                'x_mask': torch.tensor(features['mask'], dtype=torch.float32),
                'normalization': features['normalization'],
                'column_names': features['column_names'],
                'timestamps': features['timestamps']
            }

            logger.info(f"数据处理成功: {tensor_data['x'].shape}")
            return tensor_data

        except Exception as e:
            logger.error(f"CSV数据处理失败: {e}", exc_info=True)
            raise ValueError(f"数据处理失败: {e}")

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        映射列名到标准格式

        Args:
            df: 原始数据框

        Returns:
            列名映射后的数据框
        """
        # 创建列名映射字典
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            mapped_name = None

            # 直接匹配
            if col in self.column_mapping:
                mapped_name = self.column_mapping[col]
            # 模糊匹配
            else:
                for key, value in self.column_mapping.items():
                    if key.lower() in col_lower or col_lower in key.lower():
                        mapped_name = value
                        break

            if mapped_name:
                rename_dict[col] = mapped_name

        # 重命名列
        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    def _extract_features(
        self,
        df: pd.DataFrame,
        normalization_override: Dict[str, Any] = None,
    ) -> Dict[str, np.ndarray]:
        """
        提取时间序列特征

        Args:
            df: 数据框

        Returns:
            特征字典
        """
        # 确保有必要的列
        if 'timestamp' not in df.columns:
            raise ValueError("数据中缺少时间戳列")

        # 获取数值特征列
        available_columns = [col for col in self.feature_columns if col in df.columns]

        if not available_columns:
            logger.warning(f"未找到特征列，将使用所有数值列")
            available_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # 提取数值特征
        values = df[available_columns].values

        # 提取时间特征
        time_features = extract_time_features(df['timestamp'])

        # 创建掩码
        mask = create_mask(values)

        # 填充缺失值
        values = fill_missing_values(values, method="forward")
        values = fill_missing_values(values, method="zero")  # 前向填充后仍有缺失的用0填充

        # 标准化数值特征
        if normalization_override:
            normalization = {
                'min': np.asarray(normalization_override['min'], dtype=float),
                'max': np.asarray(normalization_override['max'], dtype=float),
                'range': np.asarray(normalization_override['range'], dtype=float),
            }
            if len(normalization['min']) != values.shape[1]:
                raise ValueError("Checkpoint normalization does not match input features")
            values_normalized = (values - normalization['min']) / normalization['range']
        else:
            values_normalized, normalization = self._normalize_features(values)

        return {
            'values': values_normalized,
            'time_features': time_features,
            'mask': mask,
            'column_names': available_columns,
            'timestamps': df['timestamp'].tolist(),
            'normalization': normalization
        }

    def _normalize_features(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        标准化特征

        Args:
            data: 输入数据

        Returns:
            (标准化后的数据, 标准化参数)
        """
        # 简单的MinMax标准化
        min_vals = data.min(axis=0, keepdims=True)
        max_vals = data.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除零

        normalized = (data - min_vals) / range_vals

        params = {
            'min': min_vals.flatten(),
            'max': max_vals.flatten(),
            'range': range_vals.flatten()
        }

        return normalized, params

    def denormalize_predictions(
        self,
        predictions: np.ndarray,
        params: Dict,
        column_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        反标准化预测结果

        Args:
            predictions: 标准化的预测值
            params: 标准化参数
            column_names: 列名

        Returns:
            反标准化后的预测结果字典
        """
        # 反标准化
        range_vals = params['range'].reshape(1, -1)
        min_vals = params['min'].reshape(1, -1)

        denormalized = predictions * range_vals + min_vals

        # 转换为字典
        result = {}
        for i, col_name in enumerate(column_names):
            if i < denormalized.shape[1]:
                result[col_name] = denormalized[:, i].tolist()

        return result

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        验证数据格式

        Args:
            df: 数据框

        Returns:
            (是否有效, 错误消息)
        """
        if df.empty:
            return False, "数据为空"

        # 检查是否有时间戳
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or '时间' in col]
        if not timestamp_cols:
            return False, "缺少时间戳列"

        # 检查是否有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return False, "缺少数值列"

        # 检查数据量
        if len(df) < 10:
            return False, "数据量过少，至少需要10条记录"

        return True, ""

    def get_statistics(self, csv_content: str) -> Dict[str, Any]:
        """
        获取数据统计信息

        Args:
            csv_content: CSV格式字符串

        Returns:
            统计信息字典
        """
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            df = self._map_columns(df)

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            stats = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'columns': df.columns.tolist(),
                'numeric_stats': {}
            }

            for col in numeric_cols:
                stats['numeric_stats'][col] = {
                    'count': int(df[col].count()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'missing': int(df[col].isna().sum())
                }

            return stats

        except Exception as e:
            logger.error(f"统计信息计算失败: {e}")
            return {'error': str(e)}
