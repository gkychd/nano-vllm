"""
RoPE (Rotary Position Embedding) 旋转位置编码实现

RoPE 原理：
- 通过旋转矩阵对 Query 和 Key 进行旋转编码
- 使得 token 之间的注意力分数包含相对位置信息
- 公式：Rotary(q, m) = q * cos(mθ) + rotate_half(q) * sin(mθ)
- 其中 m 是位置，θ 是频率

数学推导：
设 q 是位置 m 处的 query 向量：
1. 将 q 分为两半：q = [q_0, q_1, ..., q_{d/2-1}]
2. 对每对 (q_{2i}, q_{2i+1}) 应用旋转：
   - 旋转后的向量：[q_{2i}*cos(θ_i*m) - q_{2i+1}*sin(θ_i*m), q_{2i}*sin(θ_i*m) + q_{2i+1}*cos(θ_i*m)]

这等价于乘以旋转矩阵：
┌ cos(θ)  -sin(θ) ┐
│                │  * [x, y]
└ sin(θ)   cos(θ) ┘

优点：
- 不需要额外的位置 embedding
- 位置信息通过旋转自然融入
- 可以在推理时处理任意长度（通过旋转计算）
"""
from functools import lru_cache  # LRU 缓存装饰器
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,      # 输入 tensor
    cos: torch.Tensor,    # cos 缓存
    sin: torch.Tensor,    # sin 缓存
) -> torch.Tensor:
    """
    应用旋转位置编码

    核心公式：
    y = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]

    其中 x 被分成两半：x = [x1, x2]

    Args:
        x: 输入 tensor，shape [..., head_dim]
        cos: cos 值，shape [batch, 1, head_dim]
        sin: sin 值，shape [batch, 1, head_dim]

    Returns:
        旋转后的 tensor

    对应 C++ 实现：
    void apply_rotary_emb(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
        int64_t batch = x.size(0);
        int64_t dim = x.size(-1);

        // 分成两半
        auto x1 = x.slice(-1, 0, dim/2);
        auto x2 = x.slice(-1, dim/2, dim);

        // 计算旋转
        auto y1 = x1 * cos - x2 * sin;
        auto y2 = x2 * cos + x1 * sin;

        // 拼接
        x = torch::cat({y1, y2}, -1);
    }
    """
    # torch.chunk: 将 tensor 分成 n 份
    # dim=-1: 沿最后一个维度（head_dim）分割
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 旋转公式的第一部分：x1 * cos - x2 * sin
    y1 = x1 * cos - x2 * sin
    # 旋转公式的第二部分：x2 * cos + x1 * sin
    y2 = x2 * cos + x1 * sin

    # 拼接回去，保持原始 dtype
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    预计算并缓存 cos 和 sin 值，避免运行时重复计算

    属性：
    - cos_sin_cache: 预计算的 [cos, sin] 缓存
      shape: [max_position, 1, rotary_dim * 2]
    """

    def __init__(
        self,
        head_size: int,                    # 每个 attention 头的维度
        rotary_dim: int,                    # 旋转编码的维度（通常等于 head_size）
        max_position_embeddings: int,       # 最大位置数
        base: float,                        # 基础频率（常用 10000）
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "RoPE 维度必须等于 head_size"

        # ===== 1. 计算频率 inversed = base^(-2i/dim) =====
        # torch.arange(0, rotary_dim, 2): [0, 2, 4, ..., rotary_dim-2]
        # 步长为 2，只取偶数位置
        # 相当于 i = 0, 1, 2, ..., rotary_dim/2 - 1
        # inv_freq = base^(-2i/rotary_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # ===== 2. 计算所有位置和频率的外积 =====
        # t: [0, 1, 2, ..., max_position-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs: [max_position, rotary_dim/2]
        # freqs[i][j] = i * inv_freq[j]
        #           = i * base^(-2j/rotary_dim)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # ===== 3. 计算 cos 和 sin =====
        cos = freqs.cos()  # [max_position, rotary_dim/2]
        sin = freqs.sin()  # [max_position, rotary_dim/2]

        # ===== 4. 缓存 [cos, sin] =====
        # cat 后 shape: [max_position, rotary_dim]
        # unsqueeze_(1) 后 shape: [max_position, 1, rotary_dim]
        # 方便后续广播：cos_sin[positions] 自动扩展
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # register_buffer: 注册为非参数 tensor
        # persistent=False: 不保存到 state_dict（因为可以重新计算）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile  # PyTorch 2.0 编译优化
    def forward(
        self,
        positions: torch.Tensor,  # 位置 tensor，shape [batch]
        query: torch.Tensor,      # Q tensor，shape [batch, num_heads, head_dim]
        key: torch.Tensor,       # K tensor，shape [batch, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：应用 RoPE 到 Q 和 K

        Args:
            positions: 位置索引，shape [batch]
            query: Query tensor
            key: Key tensor

        Returns:
            应用旋转编码后的 query 和 key
        """
        # ===== 1. 取出对应位置的 cos 和 sin =====
        # positions: [batch] → [batch, 1, head_dim]
        # 自动广播：cos_sin_cache[positions] 根据 positions 取对应行
        cos_sin = self.cos_sin_cache[positions]

        # 沿最后一个维度分成两半：cos 和 sin
        # chunk(2, dim=-1) → [cos, sin]
        # 每个 shape: [batch, 1, head_dim/2]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # ===== 2. 应用旋转编码 =====
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


@lru_cache(1)  # LRU 缓存：最多缓存 1 个结果
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """
    获取 RoPE 模块（带缓存）

    使用 @lru_cache 缓存已创建的 RoPE 模块
    避免重复创建相同的模块

    对应 C++ 单例模式：
    std::shared_ptr<RotaryEmbedding> get_rope(...) {
        static auto instance = std::make_shared<RotaryEmbedding>(...);
        return instance;
    }
    """
    assert rope_scaling is None, "暂不支持 rope_scaling"
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
