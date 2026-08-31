"""
配置管理模块
"""
import os
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """应用配置"""

    # 服务配置
    app_name: str = "AquaMind Pro Backend"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000

    # APN模型配置
    apn_model_path: str = str(BACKEND_DIR / "static" / "model_weights" / "apn_water_model.pth")
    apn_model_enabled: bool = True
    apn_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LLM配置
    zhipuai_api_key: str = ""
    zhipuai_model: str = "glm-4-flash"
    llm_enabled: bool = True
    llm_fallback_enabled: bool = True

    # 降级配置
    fallback_enabled: bool = True
    fallback_mode: str = "statistical"  # statistical, rule_based

    # 数据处理配置
    max_sequence_length: int = 96
    prediction_horizon: int = 3
    feature_columns: List[str] = ["turbidity", "ph", "chlorine", "cod", "ammonia"]

    # 水务数据列名映射
    column_mapping: dict = {
        '时间': 'timestamp',
        'timestamp': 'timestamp',
        'time': 'timestamp',
        '浊度': 'turbidity',
        'turbidity': 'turbidity',
        'ntu': 'turbidity',
        'pH': 'ph',
        'ph': 'ph',
        '余氯': 'chlorine',
        'chlorine': 'chlorine',
        'COD': 'cod',
        'cod': 'cod',
        '氨氮': 'ammonia',
        'ammonia': 'ammonia',
        '溶解氧': 'do',
        'do': 'do',
        '流量': 'flow',
        'flow': 'flow',
        '加药量': 'dosage',
        'dosage': 'dosage',
        '能耗': 'energy',
        'energy': 'energy',
        '设备健康度': 'health',
        'health': 'health'
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()
