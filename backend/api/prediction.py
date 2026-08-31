"""
预测API模块 - 新增的APN预测接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import datetime

from services.container import prediction_service
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 初始化预测服务


class WaterQualityRequest(BaseModel):
    """水质预测请求模型"""
    csv_data: str
    prediction_horizon: int = 3
    target_columns: Optional[List[str]] = None


class DeviceFailureRequest(BaseModel):
    """设备故障预测请求模型"""
    device_id: str
    historical_data: Dict[str, List[float]]
    prediction_horizon: int = 24


@router.post("/water_quality")
async def predict_water_quality(request: WaterQualityRequest) -> Dict:
    """
    水质预测接口

    Args:
        request: 水质预测请求

    Returns:
        预测结果
    """
    try:
        result = prediction_service.predict_water_quality(
            csv_data=request.csv_data,
            prediction_horizon=request.prediction_horizon,
            target_columns=request.target_columns
        )

        logger.info(f"水质预测完成: 设备={request.target_columns}, 时长={request.prediction_horizon}")
        return result

    except Exception as e:
        logger.error(f"水质预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"水质预测失败: {str(e)}")


@router.post("/device_failure")
async def predict_device_failure(request: DeviceFailureRequest) -> Dict:
    """
    设备故障预测接口

    Args:
        request: 设备故障预测请求

    Returns:
        故障预测结果
    """
    try:
        result = prediction_service.predict_device_failure(
            device_id=request.device_id,
            historical_data=request.historical_data,
            prediction_horizon=request.prediction_horizon
        )

        logger.info(f"设备故障预测完成: 设备={request.device_id}")
        return result

    except Exception as e:
        logger.error(f"设备故障预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"设备故障预测失败: {str(e)}")


@router.get("/model_info")
async def get_model_info() -> Dict:
    """
    获取预测模型信息

    Returns:
        模型信息字典
    """
    return prediction_service.get_model_info()


@router.post("/energy_consumption")
async def predict_energy_consumption(request: WaterQualityRequest) -> Dict:
    """
    能耗预测接口

    Args:
        request: 能耗预测请求（复用水质预测格式）

    Returns:
        能耗预测结果
    """
    try:
        # 解析CSV数据获取历史能耗
        import pandas as pd
        import io
        from services import FallbackService

        fallback_service = FallbackService()

        df = pd.read_csv(io.StringIO(request.csv_data))
        energy_col = None

        # 查找能耗列
        for col in df.columns:
            if '能耗' in col or 'energy' in col.lower() or 'power' in col.lower():
                energy_col = col
                break

        if energy_col is None:
            # 使用第一数值列
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                energy_col = numeric_cols[0]

        if energy_col is None:
            raise HTTPException(status_code=400, detail="未找到能耗数据列")

        historical_data = df[energy_col].dropna().tolist()

        # 使用降级服务进行预测
        result = fallback_service.energy_consumption_fallback(
            historical_data=historical_data,
            prediction_horizon=request.prediction_horizon
        )

        # 添加时间戳
        base_time = datetime.datetime.now()
        result['prediction_mode'] = 'fallback'
        result['timestamps'] = [
            (base_time + datetime.timedelta(hours=i+1)).isoformat()
            for i in range(request.prediction_horizon)
        ]

        logger.info(f"能耗预测完成: 时长={request.prediction_horizon}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"能耗预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"能耗预测失败: {str(e)}")
