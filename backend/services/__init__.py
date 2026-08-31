"""
服务模块初始化
"""
from .llm_service import LLMChatService
from .prediction_service import PredictionService
from .fallback_service import FallbackService

__all__ = ['LLMChatService', 'PredictionService', 'FallbackService']