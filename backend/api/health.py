"""
健康检查API模块
"""
from fastapi import APIRouter
from typing import Dict
import datetime
import sys
import torch

from config import settings
from services import LLMChatService, FallbackService
from services.container import prediction_service
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 初始化服务
llm_service = LLMChatService()
fallback_service = FallbackService()


@router.get("/health")
async def health_check() -> Dict:
    """
    健康检查接口

    Returns:
        系统健康状态
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': settings.app_version,
        'services': {}
    }

    # 检查Python环境
    health_status['services']['python'] = {
        'status': 'healthy',
        'version': sys.version.split()[0]
    }

    # 检查PyTorch
    try:
        health_status['services']['pytorch'] = {
            'status': 'healthy',
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    except Exception as e:
        health_status['services']['pytorch'] = {
            'status': 'unhealthy',
            'error': str(e)
        }

    # 检查APN模型
    try:
        model_info = prediction_service.get_model_info()
        health_status['services']['apn_model'] = {
            'status': 'loaded' if model_info['model_loaded'] else 'fallback',
            'model_loaded': model_info['model_loaded'],
            'fallback_available': model_info['fallback_available']
        }
    except Exception as e:
        health_status['services']['apn_model'] = {
            'status': 'error',
            'error': str(e)
        }

    # 检查LLM服务
    try:
        llm_info = llm_service.get_service_info()
        health_status['services']['llm'] = {
            'status': 'enabled' if llm_info['enabled'] else 'disabled',
            'enabled': llm_info['enabled'],
            'fallback_enabled': llm_info['fallback_enabled'],
            'api_configured': llm_info['api_configured']
        }
    except Exception as e:
        health_status['services']['llm'] = {
            'status': 'error',
            'error': str(e)
        }

    # 检查降级服务
    try:
        fallback_info = fallback_service.get_service_info()
        health_status['services']['fallback'] = {
            'status': 'enabled' if fallback_info['enabled'] else 'disabled',
            'enabled': fallback_info['enabled'],
            'mode': fallback_info['mode']
        }
    except Exception as e:
        health_status['services']['fallback'] = {
            'status': 'error',
            'error': str(e)
        }

    return health_status


@router.get("/health/detailed")
async def detailed_health_check() -> Dict:
    """
    详细健康检查接口

    Returns:
        详细的系统健康状态
    """
    health = await health_check()

    # 添加系统资源信息（如果psutil可用）
    try:
        import psutil
        health['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if sys.platform != 'win32' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        health['system'] = {
            'status': 'psutil_not_available',
            'message': '安装psutil以获取系统资源信息'
        }

    # 添加配置信息
    health['config'] = {
        'app_name': settings.app_name,
        'debug_mode': settings.debug,
        'max_sequence_length': settings.max_sequence_length,
        'prediction_horizon': settings.prediction_horizon,
        'feature_columns': settings.feature_columns
    }

    return health
