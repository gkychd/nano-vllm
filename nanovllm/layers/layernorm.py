"""
RMSNorm (Root Mean Square Layer Normalization) 实现

RMSNorm 原理：
- 不同于 LayerNorm 计算均值和方差，RMSNorm 只计算 RMS（均方根）
- 公式：RMSNorm(x) = x / RMS(x) * weight
- 其中 RMS(x) = sqrt(mean(x^2))
- 优点：比 LayerNorm 更简单，计算更快，效果相当

对比 LayerNorm：
- LayerNorm: (x - mean) / sqrt(var + eps) * weight
- RMSNorm: x / sqrt(mean(x^2) + eps) * weight

减少的计算：
- 不需要计算 mean
- 不需要减去均值
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMSNorm 实现

    属性：
    - weight: 可学习的缩放参数，shape [hidden_size]
    - eps: 防止除零的小常数

    对应 C++ 实现：
    class RMSNorm {
        float eps;
        torch::Tensor weight;

        torch::Tensor forward(torch::Tensor x) {
            // 计算 RMS
            auto rms = x.pow(2).mean(-1, true).sqrt().add(eps);
            // 归一化并乘以权重
            return x / rms * weight;
        }
    };
    """

    def __init__(
        self,
        hidden_size: int,  # 隐藏层维度
        eps: float = 1e-6,  # epsilon，防止除零
    ) -> None:
        super().__init__()
        self.eps = eps
        # 初始化为 1，与 PyTorch LayerNorm 默认行为一致
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile  # PyTorch 2.0 编译优化
    def rms_forward(
        self,
        x: torch.Tensor,  # 输入 tensor，shape [..., hidden_size]
    ) -> torch.Tensor:
        """
        标准 RMSNorm 前向传播

        步骤：
        1. 保存原始 dtype
        2. 转为 float 计算（精度更高）
        3. 计算 x^2 的均值
        4. 计算 rsqrt(1/rms) = rms^(-0.5)
        5. 乘以权重并恢复 dtype

        对应公式：
        output = x * weight / sqrt(mean(x^2) + eps)
        """
        orig_dtype = x.dtype
        x = x.float()

        # 计算方差（只有 x^2 的均值）
        # dim=-1: 沿最后一维（hidden_size）计算
        # keepdim=True: 保持维度，便于广播
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt: 逆平方根 = 1/sqrt(x)
        # x / sqrt(var + eps) = x * rsqrt(var + eps)
        x.mul_(torch.rsqrt(var + self.eps))

        # 乘以可学习权重，并恢复原始 dtype
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,          # 输入
        residual: torch.Tensor,   # 残差（用于 fused 操作）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合的 RMSNorm + 残差计算

        目的：减少 kernel 启动开销，融合多个操作

        等价于：
        1. hidden = x + residual
        2. output = RMSNorm(hidden)
        3. return output, hidden（作为下次残差）

        用于 Decoder 层中：hidden_states, residual = norm(hidden_states, residual)
        """
        orig_dtype = x.dtype

        # 1. 残差相加（先转为 float 避免精度损失）
        x = x.float().add_(residual.float())
        # 保存 residual（用于后续残差连接）
        residual = x.to(orig_dtype)

        # 2. RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入 tensor
            residual: 残差（可选，用于 fused 操作）

        Returns:
            - 如果有 residual: (output, residual)
            - 如果无 residual: output
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
