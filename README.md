# AquaMind 智慧水务源码包

该目录是从开发工作区整理出的独立源码项目，不包含 Electron/Chromium 打包运行时、历史备份、日志、缓存或测试临时文件。

## 目录

- `frontend/`：智慧水务 Web 前端源码与本地静态资源。
- `backend/`：FastAPI 后端、预测服务和模型权重目录。
- `backend/apn_runtime/`：APN 推理所需的最小运行时代码。
- `backend/static/model_weights/`：放置 APN 权重文件。
- `setup.bat`：首次创建 Python 环境并安装依赖。
- `start.bat`：同时启动前端和后端。

## Windows 使用方法

1. 安装 Python 3.10 或更高版本。
2. 首次运行 `setup.bat`。
3. 按需编辑 `backend/.env`。
4. 运行 `start.bat`。
5. 浏览器会打开 `http://127.0.0.1:8080/login.html`。

后端地址为 `http://127.0.0.1:5000`，接口文档为 `http://127.0.0.1:5000/docs`。

## APN 模型

默认权重路径：

`backend/static/model_weights/apn_water_model.pth`

## APN training

The project includes `training/train_apn_water.py`, which performs chronological
data loading, train-only normalization, sliding-window construction, APN
training, validation selection, test evaluation, and checkpoint deployment.

```bat
.venv\Scripts\python.exe training\train_apn_water.py ^
  --data path\to\water_quality.csv ^
  --epochs 20 ^
  --train-stride 2
```

没有权重时，预测接口会明确返回 `prediction_mode: "fallback"`；成功加载兼容权重后返回 `prediction_mode: "apn"`。
