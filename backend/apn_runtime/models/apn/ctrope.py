import math
import torch
import torch.nn as nn

class CTRoPE(nn.Module):
    """
    Continuous-Time Rotary Positional Encoding
    将连续时间戳直接编码为 query/key 的旋转矩阵
    
    Args:
        d_model: 模型维度，必须为偶数
        omega_base_init: 基频初始值，建议 100.0（针对归一化时间窗口 [0, 1]）
        learnable: 是否将 omega_base 设为可学习参数
    """
    def __init__(self, d_model, omega_base_init=100.0, learnable=True):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        
        if learnable:
            self.log_omega_base = nn.Parameter(
                torch.tensor(math.log(omega_base_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_omega_base', 
                torch.tensor(math.log(omega_base_init), dtype=torch.float32)
            )
        
        i = torch.arange(0, d_model // 2, dtype=torch.float32)
        self.register_buffer('dim_idx', i)
    
    def forward(self, x, t):
        """
        Args:
            x: [B, M, P, D] 或 [B, P, D] 或 [1, M, 1, D] - 输入特征张量（query 或 key）
            t: [B, M, P] 或 [B, P] 或 [1, M, 1] - 真实时间戳，已归一化到 [0, 1] 或 [0, T_max]
        
        Returns:
            x_rot: 与 x 同形状的旋转后特征
        """
        omega_base = torch.exp(self.log_omega_base)
        # omega: [D/2]
        omega = omega_base ** (-2.0 * self.dim_idx / self.d_model)
        
        # 兼容 t 的不同维度数：自动 unsqueeze 到与 x 对齐
        while t.dim() < x.dim() - 1:
            t = t.unsqueeze(-1)
        theta = t.unsqueeze(-1) * omega  # [..., D/2]
        
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        # 交错重组
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_rot
    
    def get_omega_base(self):
        """诊断用：返回当前 omega_base 数值"""
        return torch.exp(self.log_omega_base).item()
