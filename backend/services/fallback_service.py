"""
降级服务模块
"""
import numpy as np
import pandas as pd
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FallbackService:
    """降级服务：当模型不可用时提供基本功能"""

    def __init__(self):
        """初始化降级服务"""
        self.enabled = settings.fallback_enabled
        self.mode = settings.fallback_mode

    def water_quality_fallback(
        self,
        csv_data: str,
        prediction_horizon: int = 3
    ) -> Dict[str, Any]:
        """
        水质预测降级：使用简单统计方法

        Args:
            csv_data: CSV数据
            prediction_horizon: 预测时间步长

        Returns:
            降级预测结果
        """
        try:
            # 解析数据
            df = pd.read_csv(io.StringIO(csv_data))

            # 简单的统计预测
            predictions = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # 使用移动平均作为简单预测
                values = df[col].dropna().values
                if len(values) > 0:
                    # 使用最后几个值的平均
                    last_values = values[-min(5, len(values)):]
                    baseline = np.mean(last_values)

                    # 添加一些随机波动
                    predictions[col] = [
                        float(baseline + np.random.normal(0, 0.1 * abs(baseline)))
                        for _ in range(prediction_horizon)
                    ]
                else:
                    predictions[col] = [0.0] * prediction_horizon

            # 生成时间戳
            timestamps = self._generate_future_timestamps(prediction_horizon)

            return {
                'prediction_mode': 'fallback',
                'predictions': predictions,
                'timestamps': timestamps,
                'confidence': 0.6,
                'method': 'statistical_fallback',
                'warning': 'APN模型未加载，使用统计方法替代',
                'model_info': {
                    'model_name': 'Statistical Fallback',
                    'model_version': '1.0.0',
                    'method': 'moving_average'
                }
            }

        except Exception as e:
            logger.error(f"降级预测失败: {e}", exc_info=True)

            # 返回默认值
            timestamps = self._generate_future_timestamps(prediction_horizon)
            return {
                'prediction_mode': 'fallback',
                'predictions': {
                    'turbidity': [2.0] * prediction_horizon,
                    'ph': [7.2] * prediction_horizon,
                    'chlorine': [0.3] * prediction_horizon
                },
                'timestamps': timestamps,
                'confidence': 0.4,
                'method': 'default_fallback',
                'warning': f'数据处理失败，使用默认值: {str(e)}',
                'error': str(e)
            }

    def device_health_fallback(
        self,
        device_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        设备健康降级：基于简单阈值

        Args:
            device_data: 设备数据字典

        Returns:
            降级健康评估结果
        """
        try:
            # 提取温度数据
            temperatures = device_data.get('temperature', [])

            if not temperatures:
                return {
                    'prediction_mode': 'fallback',
                    'health_score': 50,
                    'failure_probability': 0.5,
                    'recommendation': '设备数据不足，无法准确评估',
                    'method': 'insufficient_data_fallback'
                }

            # 计算平均温度
            avg_temp = np.mean(temperatures)
            max_temp = np.max(temperatures)
            temp_trend = np.polyfit(range(len(temperatures)), temperatures, 1)[0]  # 温度趋势

            # 基于温度评估健康度
            if max_temp > 70 or avg_temp > 65:
                # 严重过热
                health_score = 30
                failure_probability = 0.8
                recommendation = "设备严重过热，立即停机检查！"
            elif max_temp > 60 or avg_temp > 55:
                # 温度偏高
                health_score = 50
                failure_probability = 0.6
                recommendation = "设备温度偏高，建议尽快安排检修。"
            elif temp_trend > 0.5:  # 温度快速上升
                health_score = 60
                failure_probability = 0.4
                recommendation = "设备温度上升趋势明显，建议密切关注。"
            elif avg_temp > 45:
                # 温度正常偏高
                health_score = 75
                failure_probability = 0.2
                recommendation = "设备运行温度正常偏高，建议按计划检查。"
            else:
                # 温度正常
                health_score = 90
                failure_probability = 0.05
                recommendation = "设备运行状态良好，继续保持。"

            # 预测故障时间
            hours_to_failure = self._estimate_hours_to_failure(health_score)
            predicted_failure_time = (datetime.now() + timedelta(hours=hours_to_failure)).isoformat()

            return {
                'prediction_mode': 'fallback',
                'health_score': health_score,
                'failure_probability': failure_probability,
                'predicted_failure_time': predicted_failure_time,
                'recommendation': recommendation,
                'current_temperature': float(avg_temp),
                'max_temperature': float(max_temp),
                'temperature_trend': float(temp_trend),
                'method': 'threshold_based_fallback',
                'warning': 'APN模型未加载，使用阈值评估方法'
            }

        except Exception as e:
            logger.error(f"设备健康降级评估失败: {e}", exc_info=True)

            return {
                'prediction_mode': 'fallback',
                'health_score': 50,
                'failure_probability': 0.5,
                'recommendation': '设备评估失败，建议人工检查',
                'method': 'error_fallback',
                'error': str(e)
            }

    def _estimate_hours_to_failure(self, health_score: float) -> int:
        """
        估计距离故障的小时数

        Args:
            health_score: 健康度分数

        Returns:
            估计的小时数
        """
        if health_score < 40:
            return 6  # 6小时内可能故障
        elif health_score < 60:
            return 24  # 24小时内可能故障
        elif health_score < 80:
            return 72  # 3天内可能故障
        else:
            return 168  # 1周内可能故障

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

    def energy_consumption_fallback(
        self,
        historical_data: List[float],
        prediction_horizon: int = 24
    ) -> Dict[str, Any]:
        """
        能耗预测降级

        Args:
            historical_data: 历史能耗数据
            prediction_horizon: 预测时间步长

        Returns:
            降级能耗预测结果
        """
        try:
            if not historical_data:
                return {
                    'prediction_mode': 'fallback',
                    'predictions': [100.0] * prediction_horizon,
                    'method': 'default_energy_fallback',
                    'warning': '无历史能耗数据'
                }

            # 使用简单移动平均
            window_size = min(7, len(historical_data))
            baseline = np.mean(historical_data[-window_size:])

            # 添加周期性模式（模拟昼夜变化）
            predictions = []
            for i in range(prediction_horizon):
                hour = (datetime.now().hour + i) % 24
                # 白天能耗高，夜晚能耗低
                hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
                predicted = baseline * hour_factor
                predictions.append(float(predicted))

            return {
                'prediction_mode': 'fallback',
                'predictions': predictions,
                'baseline': float(baseline),
                'method': 'periodic_fallback',
                'warning': 'APN模型未加载，使用周期性模式预测'
            }

        except Exception as e:
            logger.error(f"能耗降级预测失败: {e}", exc_info=True)
            return {
                'prediction_mode': 'fallback',
                'predictions': [100.0] * prediction_horizon,
                'method': 'error_energy_fallback',
                'error': str(e)
            }

    def is_enabled(self) -> bool:
        """检查降级服务是否启用"""
        return self.enabled

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'enabled': self.enabled,
            'mode': self.mode,
            'available_methods': [
                'water_quality_fallback',
                'device_health_fallback',
                'energy_consumption_fallback'
            ]
        }
