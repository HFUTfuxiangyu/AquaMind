"""
AquaMind Pro 后端服务主入口
"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 添加项目路径
backend_dir = Path(__file__).parent
apn_dir = backend_dir / 'apn_runtime'
if apn_dir.exists():
    sys.path.append(str(apn_dir))

from config import settings
from api import chat_router, prediction_router, health_router
from services import LLMChatService, FallbackService
from services.container import prediction_service
from utils.logger import setup_logger, get_logger

# 设置日志
logger = setup_logger(
    name="AquaMind",
    level="DEBUG" if settings.debug else "INFO",
    log_file=str(backend_dir / "logs" / "app.log") if not settings.debug else None
)

# 初始化服务
llm_service = LLMChatService()
fallback_service = FallbackService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")
    logger.info(f"运行模式: {'开发' if settings.debug else '生产'}")
    logger.info(f"监听地址: {settings.host}:{settings.port}")

    # 尝试加载APN模型
    if settings.apn_model_enabled:
        try:
            model_path = Path(settings.apn_model_path)
            if model_path.exists():
                success = prediction_service.load_apn_model(str(model_path))
                if success:
                    logger.info("APN模型加载成功")
                else:
                    logger.warning("APN模型加载失败，将使用降级服务")
            else:
                logger.warning(f"模型权重文件不存在: {settings.apn_model_path}")
                logger.info("系统将在降级模式下运行")
        except Exception as e:
            logger.warning(f"APN模型初始化异常: {e}")
            logger.info("系统将在降级模式下运行")
    else:
        logger.info("APN模型功能已禁用")

    # 检查LLM服务状态
    if llm_service.is_enabled():
        logger.info("LLM聊天服务已启用")
    else:
        logger.info("LLM聊天服务未启用，将使用降级响应")

    # 检查降级服务
    if fallback_service.is_enabled():
        logger.info("降级服务已启用")

    logger.info("服务启动完成，准备接受请求")

    yield

    # 关闭时清理
    logger.info("正在关闭服务...")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="智慧水务AI大脑后端服务 - 基于APN模型和LLM的智能水务管理系统",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部服务器错误",
            "error": str(exc) if settings.debug else "请联系管理员"
        }
    )


# 注册路由
app.include_router(chat_router, prefix="/api", tags=["聊天服务"])
app.include_router(prediction_router, prefix="/api/predict", tags=["预测服务"])
app.include_router(health_router, prefix="/api", tags=["健康检查"])


# 根路径
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "智慧水务AI大脑后端服务",
        "status": "running",
        "endpoints": {
            "/api/chat": "AI对话接口（兼容原有格式）",
            "/api/predict/water_quality": "水质预测接口（新增）",
            "/api/predict/device_failure": "设备故障预测接口（新增）",
            "/api/predict/energy_consumption": "能耗预测接口（新增）",
            "/api/health": "健康检查接口",
            "/docs": "API文档（Swagger UI）",
            "/redoc": "API文档（ReDoc）"
        },
        "services": {
            "apn_model": "loaded" if prediction_service.model_loaded else "fallback",
            "llm": "enabled" if llm_service.is_enabled() else "disabled",
            "fallback": "enabled" if fallback_service.is_enabled() else "disabled"
        },
        "documentation": "https://github.com/your-repo/AquaMind-Pro"
    }


# 健康检查快捷方式
@app.get("/health")
async def health_check_redirect():
    """健康检查重定向"""
    from api.health import health_check
    return await health_check()


def main():
    """主函数"""
    # 创建日志目录
    log_dir = backend_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型权重目录
    model_dir = backend_dir / "static" / "model_weights"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 运行服务器
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True
    )


if __name__ == "__main__":
    main()
