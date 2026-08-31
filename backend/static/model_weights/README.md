# APN 模型权重

将与当前后端配置兼容的水质 APN 权重放在此目录，并命名为：

`apn_water_model.pth`

权重未提供时，后端会自动使用降级预测，并返回 `prediction_mode: "fallback"`。
