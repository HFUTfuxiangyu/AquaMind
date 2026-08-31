"""
LLM聊天服务模块
"""
import os
import httpx
import json
from typing import List, Dict, Optional
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMChatService:
    """LLM聊天服务，兼容原有API"""

    def __init__(self):
        """初始化LLM聊天服务"""
        self.api_key = settings.zhipuai_api_key
        self.model = settings.zhipuai_model
        self.enabled = settings.llm_enabled
        self.fallback_enabled = settings.llm_fallback_enabled

        # 智谱AI API配置
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

        if not self.api_key and self.enabled:
            logger.warning("未设置ZHIPUAI_API_KEY，将使用降级服务")
            self.enabled = False

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        处理聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            AI回复内容
        """
        if not self.enabled:
            if self.fallback_enabled:
                logger.info("LLM服务未启用，使用降级响应")
                return self._fallback_response(messages)
            else:
                raise RuntimeError("LLM服务未启用且降级服务不可用")

        try:
            response = await self._call_zhipu_api(messages, temperature, max_tokens)
            return response

        except Exception as e:
            logger.error(f"LLM调用失败: {e}", exc_info=True)
            if self.fallback_enabled:
                logger.info("LLM调用失败，使用降级响应")
                return self._fallback_response(messages)
            else:
                raise

    async def _call_zhipu_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        调用智谱AI API

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            API响应内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()

            # 提取回复内容
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise ValueError("API响应格式不正确")

    def _fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """
        降级响应：基于规则的简单回复

        Args:
            messages: 消息列表

        Returns:
            规则匹配的回复
        """
        if not messages:
            return "我是智慧水务AI助手，请问有什么可以帮您？"

        last_message = messages[-1]['content'].lower()

        # 规则匹配
        rules = [
            (['水质', '浊度', 'ph', '余氯', 'cod', '氨氮'],
             "当前水质监测系统运行正常，主要指标均在标准范围内。浊度、pH值、余氯等关键参数保持稳定。建议继续关注进水水质变化，及时调整处理工艺。"),

            (['设备', '故障', '维护', '检修', '泵', '阀'],
             "设备监控显示关键设备运行状态良好。各主要设备均在正常工作参数范围内运行。如需详细的设备状态信息，请查看设备健康监控页面。建议按计划进行预防性维护。"),

            (['加药', '药剂', '混凝剂', 'pac', '投加'],
             "加药系统根据当前水质参数自动调节投加量。系统会根据进水浊度、pH值等指标优化药剂投加策略。建议查看AI加药优化页面获取详细的加药建议和成本分析。"),

            (['能耗', '电费', '节能', '峰谷', '电价'],
             "系统能耗运行在正常范围内。通过峰谷电价优化和设备调度，可实现一定的节能效果。建议查看能效调度页面了解详细的能耗分析和优化建议。"),

            (['报告', '分析', '统计', '数据'],
             "系统可以为您生成详细的水务运营分析报告，包括水质指标趋势、设备运行状态、能耗分析等内容。请告诉我您需要哪种类型的报告，我会为您生成相应的分析内容。"),

            (['预测', '预报', '预警', '趋势'],
             "基于历史数据和AI模型，系统可以预测未来几小时的水质变化趋势、设备运行状态等。这有助于提前发现潜在问题并采取相应措施。预测功能正在持续优化中。"),

            (['帮助', '使用', '功能'],
             "我可以帮助您分析水务数据、提供设备运行建议、优化加药策略、生成运营报告等。主要功能包括：水质监测分析、设备健康管理、智能加药优化、能效调度分析等。请告诉我您的具体需求。")
        ]

        # 匹配规则
        for keywords, response in rules:
            if any(keyword in last_message for keyword in keywords):
                return response

        # 默认回复
        return "我是智慧水务AI助手，可以帮您分析水质数据、设备状态、优化加药策略、生成运营报告等。请问有什么具体可以帮您的？"

    def is_enabled(self) -> bool:
        """检查服务是否启用"""
        return self.enabled

    def get_service_info(self) -> Dict[str, any]:
        """获取服务信息"""
        return {
            'enabled': self.enabled,
            'model': self.model,
            'fallback_enabled': self.fallback_enabled,
            'api_configured': bool(self.api_key)
        }