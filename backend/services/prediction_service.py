"""
预测服务模块
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from models import APNModelWrapper, WaterDataProcessor, ModelNotLoadedError
# 延迟导入以避免循环依赖
# from services import FallbackService
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """预测服务"""

    def __init__(self):
        """初始化预测服务"""
        self.model_wrapper = APNModelWrapper()
        self.data_processor = WaterDataProcessor()
        self._fallback_service = None  # 延迟初始化
        self.model_loaded = False

    @property
    def fallback_service(self):
        """延迟导入FallbackService以避免循环依赖"""
        if self._fallback_service is None:
            from services.fallback_service import FallbackService
            self._fallback_service = FallbackService()
        return self._fallback_service

    def load_apn_model(self, model_path: str) -> bool:
        """
        加载APN模型

        Args:
            model_path: 模型权重路径

        Returns:
            是否加载成功
        """
        success = self.model_wrapper.load_model(model_path)
        self.model_loaded = success
        return success

    def predict_water_quality(
        self,
        csv_data: str,
        prediction_horizon: int = None,
        target_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        水质预测

        Args:
            csv_data: CSV格式的水质数据
            prediction_horizon: 预测时间步长
            target_columns: 目标预测列

        Returns:
            预测结果
        """
        if prediction_horizon is None:
            prediction_horizon = settings.prediction_horizon

        if target_columns is None:
            target_columns = settings.feature_columns

        try:
            # 如果模型未加载，使用降级服务
            if not self.model_loaded:
                logger.info("APN模型未加载，使用降级服务")
                return self.fallback_service.water_quality_fallback(csv_data, prediction_horizon)

            # 数据预处理
            tensor_data = self.data_processor.csv_to_tensor(
                csv_data,
                normalization_override=self.model_wrapper.get_normalization(),
            )

            # 执行预测
            predictions = self.model_wrapper.predict(tensor_data, prediction_horizon=prediction_horizon)
            if 'predictions' in predictions and 'normalization' in tensor_data:
                pred_array = np.array(predictions['predictions'])
                predictions['predictions'] = self.data_processor.denormalize_predictions(
                    pred_array,
                    tensor_data['normalization'],
                    tensor_data['column_names']
                )

            # 生成未来时间戳
            future_timestamps = self._generate_future_timestamps(
                prediction_horizon
            )

            # 提取目标列的预测结果
            result = self._extract_target_predictions(
                predictions,
                target_columns,
                future_timestamps
            )

            # 添加模型信息
            result['model_info'] = predictions.get('model_info', {})
            result['prediction_mode'] = 'apn'
            result['confidence'] = 0.85  # 默认置信度
            result['prediction_horizon'] = prediction_horizon

            logger.info(f"水质预测完成: {len(result.get('predictions', {}))}个参数")
            return result

        except ModelNotLoadedError:
            logger.warning("模型加载失败，使用降级服务")
            return self.fallback_service.water_quality_fallback(csv_data, prediction_horizon)

        except Exception as e:
            logger.error(f"水质预测失败: {e}", exc_info=True)
            return self.fallback_service.water_quality_fallback(csv_data, prediction_horizon)

    def predict_device_failure(
        self,
        device_id: str,
        historical_data: Dict[str, List[float]],
        prediction_horizon: int = 24
    ) -> Dict[str, Any]:
        """
        设备故障预测

        Args:
            device_id: 设备ID
            historical_data: 历史数据字典
            prediction_horizon: 预测时间范围（小时）

        Returns:
            故障预测结果
        """
        try:
            # 如果模型未加载，使用降级服务
            if not self.model_loaded:
                logger.info("APN模型未加载，使用降级服务")
                return self.fallback_service.device_health_fallback(historical_data)

            # 数据预处理
            tensor_data = self._prepare_device_data(historical_data)

            # 执行预测
            predictions = self.model_wrapper.predict(tensor_data, prediction_horizon=prediction_horizon)

            # 分析预测结果
            result = self._analyze_device_predictions(
                device_id,
                predictions,
                historical_data,
                prediction_horizon
            )

            logger.info(f"设备故障预测完成: {device_id}")
            return result

        except Exception as e:
            logger.error(f"设备故障预测失败: {e}", exc_info=True)
            return self.fallback_service.device_health_fallback(historical_data)

    def _prepare_device_data(self, historical_data: Dict[str, List[float]]) -> Dict[str, torch.Tensor]:
        """
        准备设备数据

        Args:
            historical_data: 历史数据

        Returns:
            张量数据
        """
        # 转换为numpy数组
        data_arrays = []
        for key in ['temperature', 'vibration', 'current', 'pressure']:
            if key in historical_data:
                values = np.array(historical_data[key]).reshape(-1, 1)
                data_arrays.append(values)

        if not data_arrays:
            raise ValueError("没有有效的设备数据")

        # 拼接所有特征
        combined_data = np.concatenate(data_arrays, axis=1)

        # 创建时间特征（简化版）
        time_features = np.tile([0.5, 0.5, 0.5, 0.5], (len(combined_data), 1))

        # 创建掩码
        mask = np.ones_like(combined_data)

        return {
            'x': torch.tensor(combined_data, dtype=torch.float32),
            'x_mark': torch.tensor(time_features, dtype=torch.float32),
            'x_mask': torch.tensor(mask, dtype=torch.float32)
        }

    def _analyze_device_predictions(
        self,
        device_id: str,
        predictions: Dict,
        historical_data: Dict,
        prediction_horizon: int
    ) -> Dict[str, Any]:
        """
        分析设备预测结果

        Args:
            device_id: 设备ID
            predictions: 预测结果
            historical_data: 历史数据
            prediction_horizon: 预测时间范围

        Returns:
            分析结果
        """
        # 提取预测值
        pred_values = predictions.get('predictions', [])
        if not pred_values:
            return self.fallback_service.device_health_fallback(historical_data)

        # 计算健康度分数
        health_score = self._calculate_health_score(pred_values, historical_data)

        # 计算故障概率
        failure_probability = max(0, (100 - health_score) / 100)

        # 预测故障时间
        predicted_failure_time = self._predict_failure_time(
            health_score,
            prediction_horizon
        )

        # 生成建议
        recommendation = self._generate_device_recommendation(health_score, failure_probability)

        return {
            'device_id': device_id,
            'health_score': health_score,
            'failure_probability': failure_probability,
            'predicted_failure_time': predicted_failure_time,
            'recommendation': recommendation,
            'predictions': pred_values,
            'model_info': predictions.get('model_info', {}),
            'prediction_mode': 'apn'
        }

    def _calculate_health_score(
        self,
        predictions: List,
        historical_data: Dict
    ) -> float:
        """
        计算设备健康度分数

        Args:
            predictions: 预测值
            historical_data: 历史数据

        Returns:
            健康度分数 (0-100)
        """
        # 简化的健康度计算
        base_score = 85.0

        # 根据预测值调整
        if predictions and len(predictions) > 0:
            # 假设第一个特征是温度
            temp_predictions = [p[0] if isinstance(p, list) else p for p in predictions[:3]]
            avg_temp = sum(temp_predictions) / len(temp_predictions)

            if avg_temp > 60:
                base_score -= 30
            elif avg_temp > 50:
                base_score -= 15
            elif avg_temp > 45:
                base_score -= 5

        return max(0, min(100, base_score))

    def _predict_failure_time(
        self,
        health_score: float,
        prediction_horizon: int
    ) -> str:
        """
        预测故障时间

        Args:
            health_score: 健康度分数
            prediction_horizon: 预测时间范围

        Returns:
            预测故障时间字符串
        """
        if health_score < 50:
            # 高风险，短期内可能故障
            hours = prediction_horizon // 2
        elif health_score < 70:
            # 中等风险
            hours = prediction_horizon
        else:
            # 低风险，长期内正常
            hours = prediction_horizon * 2

        failure_time = datetime.now() + timedelta(hours=hours)
        return failure_time.isoformat()

    def _generate_device_recommendation(
        self,
        health_score: float,
        failure_probability: float
    ) -> str:
        """
        生成设备维护建议

        Args:
            health_score: 健康度分数
            failure_probability: 故障概率

        Returns:
            建议文本
        """
        if health_score < 50:
            return f"设备健康度较低({health_score:.0f}分)，故障概率较高({failure_probability:.1%})，建议立即安排检修。"
        elif health_score < 70:
            return f"设备健康度一般({health_score:.0f}分)，建议在{24}小时内安排检查。"
        elif health_score < 85:
            return f"设备运行基本正常({health_score:.0f}分)，建议按计划进行预防性维护。"
        else:
            return f"设备运行状态良好({health_score:.0f}分)，继续保持当前运行状态。"

    def _generate_future_timestamps(self, hours: int) -> List[str]:
        """
        生成未来时间戳

        Args:
            hours: 小时数

        Returns:
            时间戳列表
        """
        timestamps = []
        base_time = datetime.now()

        for i in range(1, hours + 1):
            future_time = base_time + timedelta(hours=i)
            timestamps.append(future_time.isoformat())

        return timestamps

    def _extract_target_predictions(
        self,
        predictions: Dict,
        target_columns: List[str],
        future_timestamps: List[str]
    ) -> Dict[str, Any]:
        """
        提取目标列的预测结果

        Args:
            predictions: 完整预测结果
            target_columns: 目标列
            future_timestamps: 未来时间戳

        Returns:
            提取后的预测结果
        """
        pred_array = predictions.get('predictions', [])
        if not pred_array:
            return {'predictions': {}, 'timestamps': future_timestamps}

        if isinstance(pred_array, dict):
            selected = {
                col: pred_array[col]
                for col in target_columns
                if col in pred_array
            }
            return {'predictions': selected, 'timestamps': future_timestamps}

        # 转换为numpy数组
        pred_values = np.array(pred_array)

        result = {'predictions': {}, 'timestamps': future_timestamps}

        # 提取每个目标列的预测
        for i, col in enumerate(target_columns):
            if i < pred_values.shape[1]:
                # 取每个时间步的预测
                col_predictions = pred_values[:, i].tolist()
                result['predictions'][col] = col_predictions

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_loaded': self.model_loaded,
            'prediction_mode': 'apn' if self.model_loaded else 'fallback',
            'model_info': self.model_wrapper.get_model_info() if self.model_loaded else None,
            'fallback_available': True
        }
