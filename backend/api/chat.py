"""
聊天API模块 - 兼容原有接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import datetime
import time

from services import LLMChatService
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 初始化聊天服务
chat_service = LLMChatService()


class Message(BaseModel):
    """消息模型"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048


class ChatResponse(BaseModel):
    """聊天响应模型 - 兼容前端期望的OpenAI格式"""
    choices: List[Dict[str, Any]]
    model: str
    created: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    AI聊天接口 - 兼容原有格式

    Args:
        request: 聊天请求

    Returns:
        聊天响应
    """
    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # 调用聊天服务
        response_text = await chat_service.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # 构建响应 - 兼容前端期望的OpenAI格式
        import time
        response = ChatResponse(
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop",
                "index": 0
            }],
            model=chat_service.model,
            created=int(time.time())
        )

        logger.info(f"聊天请求处理成功: {len(messages)}条消息")
        return response

    except Exception as e:
        logger.error(f"聊天请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"聊天服务错误: {str(e)}")


@router.get("/chat/info")
async def get_chat_info() -> Dict:
    """
    获取聊天服务信息

    Returns:
        服务信息字典
    """
    return chat_service.get_service_info()