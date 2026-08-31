# AquaMind APN 水质训练

`train_apn_water.py` 是 AquaMind 自主训练入口，负责时间切分读取、仅基于训练集的归一化、滑动窗口构造、APN 训练、验证集选模、测试集评估和 checkpoint 部署。

输入 CSV 字段：

`timestamp,turbidity,ph,chlorine,cod,ammonia,split`

其中 `split` 必须是 `train`、`validation` 或 `test`。训练脚本不会随机重划分数据，避免时间序列泄漏。

默认部署位置：

`backend/static/model_weights/apn_water_model.pth`

配套指标文件：

`backend/static/model_weights/apn_water_model.metrics.json`
