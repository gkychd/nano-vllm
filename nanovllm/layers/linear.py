"""
Linear 层实现（张量并行）

核心概念：张量并行（Tensor Parallelism）
- 将模型的权重矩阵分割到多个 GPU 上
- 每个 GPU 只存储和计算部分权重
- 通过通信原语（all_reduce）汇总结果

张量并行示意图（2 GPU，输出维度 4096）：
                    全连接层
              ┌─────────────────┐
              │    W [4096x4096] │
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
     ┌─────▼─────┐           ┌─────▼─────┐
     │W0 [2048x4096]│         │W1 [2048x4096]│
     │ GPU 0     │           │ GPU 1     │
     └─────┬─────┘           └─────┬─────┘
           │                       │
     ┌─────▼─────┐           ┌─────▼─────┐
     │y0 [bsx2048]│           │y1 [bsx2048]│
     └─────┬─────┘           └─────┬─────┘
           │                       │
           └───────────┬───────────┘
                       │
              ┌────────▼────────┐
              │ all_reduce(y)   │
              │ y = y0 + y1     │
              └─────────────────┘
                       │
              ┌────────▼────────┐
              │   y [bsx4096]   │
              └─────────────────┘
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除检查，确保能整除"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    Linear 层基类

    属性说明：
    - tp_dim: 张量并行分割的维度
      - None: 不分割（复制）
      - 0: 按输出维度分割（列并行）
      - 1: 按输入维度分割（行并行）
    - tp_rank: 当前进程的编号（0, 1, 2, ...）
    - tp_size: 总进程数（GPU 数量）

    对应 C++ 结构体：
    struct LinearBase {
        int tp_dim;      // 张量并行维度
        int tp_rank;     // 当前进程 rank
        int tp_size;     // 总进程数
        torch::Tensor weight;  // 权重矩阵
        torch::Tensor bias;    // 偏置（可选）
    };
    """

    def __init__(
        self,
        input_size: int,      # 输入维度
        output_size: int,     # 输出维度
        bias: bool = False,   # 是否使用偏置
        tp_dim: int | None = None,  # 张量并行维度
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()      # 获取当前进程 rank
        self.tp_size = dist.get_world_size()  # 获取总进程数

        # 创建权重参数
        # nn.Parameter: 自动注册为模型参数，会被 optimizer 更新
        # shape: [output_size, input_size]（PyTorch 惯例）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        # 设置自定义 weight_loader（用于分布式加载权重）
        self.weight.weight_loader = self.weight_loader

        # 偏置处理
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # register_parameter 创建空参数（值为 None）
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """子类必须实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层（Replicated Linear）

    特点：权重在所有 GPU 上完全复制一份
    适用场景：不需要张量并行的层

    内存占用：所有 GPU 相同（完整权重）
    通信开销：无（无需通信）

    对应 C++：
    class ReplicatedLinear {
        torch::Tensor weight;  // 完整权重，所有 GPU 相同

        torch::Tensor forward(torch::Tensor x) {
            return torch::mm(x, weight.t());
        }
    };
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器

        Args:
            param: 目标参数
            loaded_weight: 从 HF 加载的完整权重

        直接复制，无需分割
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear 是 torch.mm + bias 的融合实现
        # y = x @ W.T + b
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层（Column Parallel Linear）

    特点：按输出维度分割权重（列切割）
    权重 shape: [output_size/tp_size, input_size]

    示意图（2 GPU，输出 4096 → 2048）：
    W = [W0; W1]  (垂直分割)
    y = x @ W.T = x @ [W0.T, W1.T]
    y0 = x @ W0.T  (GPU 0 计算)
    y1 = x @ W1.T  (GPU 1 计算)
    y = [y0, y1]   (拼接结果)

    适用场景：
    - MLP 的第一个线性层（up_proj）
    - Attention 的 QKV 投影

    对应 C++：
    class ColumnParallelLinear {
        torch::Tensor weight;  // [output_size/tp_size, input_size]

        torch::Tensor forward(torch::Tensor x) {
            // 每个 GPU 计算部分输出
            return torch::mm(x, weight.t());
        }
    };
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输出维度除以 tp_size，每个 GPU 只计算一部分
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重分片加载

        从完整权重中切出当前 rank 对应的部分

        示例（2 GPU，output_size=4096）：
        - GPU 0: loaded_weight.narrow(0, 0, 2048)  → [2048, 4096]
        - GPU 1: loaded_weight.narrow(0, 2048, 2048) → [2048, 4096]

        narrow(dim, start, length): 沿指定维度切片
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # 每个分片的大小
        start_idx = self.tp_rank * shard_size  # 当前 rank 的起始位置
        # 切出对应分片
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 ReplicatedLinear 相同，前向计算逻辑不变
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层

    特点：多个线性层合并为一个，共享输入
    适用场景：MLP 中 up_proj + gate_proj 合并

    示例：
    - 输入 x: [batch, hidden_size]
    - up_proj: [hidden_size, 4*hidden_size]
    - gate_proj: [hidden_size, 4*hidden_size]
    - 合并后: [hidden_size, 8*hidden_size]

    每个 GPU 计算：
    - up: x @ up_weight.T
    - gate: x @ gate_weight.T
    - 输出: [up, gate] 拼接
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # 多个输出的维度
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出维度求和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        加载指定分片的权重

        Args:
            loaded_shard_id: 要加载的输出索引（0, 1, 2, ...）

        示例：
        output_sizes = [4096, 4096]，tp_size = 2
        - loaded_shard_id = 0: up_proj
          GPU 0: offset=0, size=2048
          GPU 1: offset=2048, size=2048
        - loaded_shard_id = 1: gate_proj
          GPU 0: offset=0, size=2048
          GPU 1: offset=2048, size=2048
        """
        param_data = param.data
        # 计算该分片在合并输出中的偏移
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 目标位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从源权重切出对应部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 并行线性层

    特点：同时计算 Query、Key、Value 三个投影
    权重 shape: [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]

    权重布局：
    ┌─────────────────────────────────────────┐
    │      Q_proj      │   K_proj   │  V_proj │
    │ num_heads*head_dim  │ num_kv*head_dim │ num_kv*head_dim │
    └─────────────────────────────────────────┘

    每个 GPU 计算：
    - Q = x @ Q_weight.T
    - K = x @ K_weight.T
    - V = x @ V_weight.T

    对应 C++：
    class QKVParallelLinear {
        torch::Tensor qkv_weight;  // 合并的 QKV 权重

        std::vector<torch::Tensor> forward(torch::Tensor x) {
            auto qkv = torch::mm(x, qkv_weight.t());
            // 分割 Q、K、V
            auto q = qkv.narrow(1, 0, num_heads * head_dim);
            auto k = qkv.narrow(1, num_heads * head_dim, num_kv_heads * head_dim);
            auto v = qkv.narrow(1, num_heads * head_dim + num_kv_heads * head_dim, num_kv_heads * head_dim);
            return {q, k, v};
        }
    };
    """

    def __init__(
        self,
        hidden_size: int,           # 输入维度
        head_size: int,              # 每个头的维度
        total_num_heads: int,        # 总 Q 头数
        total_num_kv_heads: int | None = None,  # 总 KV 头数
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每个 GPU 的 Q 头数
        self.num_heads = divide(total_num_heads, tp_size)
        # 每个 GPU 的 KV 头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出维度 = Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        加载 QKV 中指定部分的权重

        Args:
            loaded_shard_id: "q", "k", 或 "v"

        权重布局：
        ┌──────────────────────────────────────────────┐
        │        Q               │       K       │   V  │
        │ num_heads*head_dim    │ num_kv*hd    │ num_kv*hd │
        │   [0, num_heads*hd)   │ [num_h*hd,   │ [最后一段]  │
        │                        │  num_h*hd+   │           │
        │                        │  num_kv*hd)  │           │
        └──────────────────────────────────────────────┘
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            # Q 在最前面
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K 在中间
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            # V 在最后
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        # 目标位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 按 tp_size 分割，取当前 rank 的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层（Row Parallel Linear）

    特点：按输入维度分割权重（行切割）
    权重 shape: [output_size, input_size/tp_size]

    示意图（2 GPU，输入 4096 → 2048）：
    W = [W0; W1]  (水平分割)
    W0: [4096, 2048], W1: [4096, 2048]

    计算过程：
    1. 输入 x 被分割：x = [x0, x1]
    2. GPU 0: y0 = x0 @ W0.T
    3. GPU 1: y1 = x1 @ W1.T
    4. all_reduce: y = y0 + y1（汇总所有 GPU 的结果）

    适用场景：
    - MLP 的第二个线性层（down_proj）

    对应 C++：
    class RowParallelLinear {
        torch::Tensor weight;  // [output_size, input_size/tp_size]

        torch::Tensor forward(torch::Tensor x) {
            // 1. 计算部分结果
            auto y = torch::mm(x, weight.t());

            // 2. 多 GPU 汇总（all_reduce）
            if (tp_size > 1) {
                dist::all_reduce(y);
            }
            return y;
        }
    };
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度除以 tp_size，每个 GPU 只计算一部分输入
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重分片加载（按行分割）

        示例（2 GPU，input_size=4096）：
        - GPU 0: loaded_weight.narrow(1, 0, 2048)  → [4096, 2048]
        - GPU 1: loaded_weight.narrow(1, 2048, 2048) → [4096, 2048]
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 线性计算（每个 GPU 只处理部分输入）
        # 只有 rank 0 添加 bias（最终 bias 只有一份）
        y = F.linear(
            x,
            self.weight,
            self.bias if self.tp_rank == 0 else None
        )

        # 2. 多 GPU 汇总（核心通信操作）
        # all_reduce: 所有 GPU 的数据求和，结果同步到每个 GPU
        # 等价于: y = sum(y_i) for all GPUs
        if self.tp_size > 1:
            dist.all_reduce(y)

        return y
