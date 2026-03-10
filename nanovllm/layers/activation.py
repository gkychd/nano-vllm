"""
激活函数实现

主要实现：
- SiluAndMul: SwiGLU 的核心操作
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU (Swish) 激活函数 + 乘法

    SwiGLU 公式：
    output = SiLU(gate) * up
    - gate: 门控信号
    - up: 上投影

    SiLU 公式：
    SiLU(x) = x * sigmoid(x)
    - 也称为 Swish

    示意图：
    ┌─────────────────────────────────────────┐
    │           输入 x [batch, 2*intermediate]│
    │                  │                       │
    │          chunk(2, -1)                   │
    │          ╱           ╲                  │
    │      gate           up                  │
    │         │            │                  │
    │      SiLU(.)      identity              │
    │         │            │                  │
    │         └─────┬──────┘                  │
    │               ▼                          │
    │        SiLU(gate) * up                  │
    └─────────────────────────────────────────┘

    对应 C++ 实现：
    class SiluAndMul {
        torch::Tensor forward(torch::Tensor x) {
            // 沿最后一维分成两半
            auto [gate, up] = x.chunk(2, -1);
            // SiLU: x * sigmoid(x)
            auto swiglu = gate * sigmoid(gate);
            // 乘以上投影
            return swiglu * up;
        }
    };
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # PyTorch 2.0 编译优化
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 tensor，shape [..., 2 * intermediate_size]
                最后一维包含 gate 和 up 两部分

        Returns:
            输出 tensor，shape [..., intermediate_size]

        计算过程：
        1. chunk: 将 x 分成 gate 和 up
        2. SiLU: gate * sigmoid(gate)
        3. multiply: SiLU(gate) * up
        """
        # 1. 沿最后一维分成两半
        # x: [batch, 2*intermediate]
        # → gate: [batch, intermediate]
        # → up: [batch, intermediate]
        x, y = x.chunk(2, -1)

        # 2. 计算 SiLU(gate) * up
        # F.silu(x) = x * sigmoid(x)
        return F.silu(x) * y
