"""
Embedding 和 LM Head 实现（张量并行版本）

Embedding 和 LM Head 的张量并行策略：
- Embedding: 按词表分割，每个 GPU 只存储部分词向量
- LM Head: 同样按词表分割

Embedding 并行示意图（2 GPU，词表 10000）：
┌─────────────────────────────────────────┐
│           词表分割                        │
│  GPU 0: 0-4999    GPU 1: 5000-9999       │
└────────────────┬────────────────────────┘
                 │
        input_ids: [100, 5000, 8000]
                 │
         ┌───────┴───────┐
         │               │
    GPU 0 处理       GPU 1 处理
    (mask 有效)     (mask 有效)
         │               │
         └───────┬───────┘
                 │
          all_reduce
                 │
         [emb_100, emb_5000, emb_8000]
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding

    特点：将词表分割到多个 GPU 上

    内存节省：每个 GPU 只存储 vocab_size / tp_size 个词向量

    对应 C++：
    class VocabParallelEmbedding {
        int tp_rank, tp_size;
        int vocab_start_idx, vocab_end_idx;
        torch::Tensor weight;  // [vocab/tp_size, embedding_dim]

        torch::Tensor forward(torch::Tensor x) {
            // 1. 映射 token 到本地索引
            auto mask = (x >= vocab_start_idx) && (x < vocab_end_idx);
            auto local_x = (x - vocab_start_idx) * mask;

            // 2. 查表
            auto y = embedding(local_x, weight);

            // 3. 乘以 mask（无效位置置零）
            y = y * mask.unsqueeze(-1);

            // 4. all_reduce 汇总
            if (tp_size > 1) {
                dist::all_reduce(y);
            }
            return y;
        }
    };
    """

    def __init__(
        self,
        num_embeddings: int,   # 词表大小
        embedding_dim: int,    # 嵌入维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        # 每个 GPU 的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 当前 GPU 负责的词表范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 权重 shape：[vocab/tp_size, embedding_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中加载对应分片"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: input_ids，shape [batch, seqlen]

        Returns:
            embeddings，shape [batch, seqlen, embedding_dim]
        """
        # ===== 1. 张量并行：映射 token 到本地索引 =====
        if self.tp_size > 1:
            # 创建 mask：标记哪些 token 属于当前 GPU
            # 例如：GPU 0 负责 0-4999，mask 标记 input_ids 在这个范围的
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 转换为本地索引（相对于当前 GPU 词表的索引）
            x = mask * (x - self.vocab_start_idx)

        # ===== 2. Embedding 查表 =====
        # F.embedding 等价于 torch.nn.functional.embedding
        # y[i] = weight[input_ids[i]]
        y = F.embedding(x, self.weight)

        # ===== 3. 张量并行：汇总结果 =====
        if self.tp_size > 1:
            # 乘以 mask（无效位置置零）
            # 有效位置：mask=1，保留原值
            # 无效位置：mask=0，结果为 0
            y = mask.unsqueeze(1) * y
            # all_reduce：所有 GPU 求和
            # 结果：每个 GPU 都有完整的 embedding
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行 LM Head

    作用：将 hidden states 转换为 logits

    特点：
    - 继承 VocabParallelEmbedding（词表分割策略相同）
    - Prefill 阶段：只计算最后一个 token 的 logits
    - Decode 阶段：计算所有 token 的 logits
    - 张量并行：使用 gather 收集所有 GPU 的结果
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: hidden states，[batch, seqlen, hidden_size] 或 [batch, hidden_size]

        Returns:
            logits: [batch, vocab_size] 或 [batch*seqlen, vocab_size]
        """
        # 获取上下文信息
        context = get_context()

        # ===== Prefill 阶段特殊处理 =====
        # Prefill 时只需要最后一个 token 的 logits（用于生成下一个 token）
        # 不需要计算整个序列的 logits，节省计算
        if context.is_prefill:
            # cu_seqlens_q: 累计序列长度
            # 取每个序列的最后一个位置的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # 只取最后一个 token 的 hidden states
            x = x[last_indices].contiguous()

        # ===== 线性变换 =====
        # x @ W.T + b（这里 b=False）
        # x: [batch, hidden_size]
        # weight: [vocab/tp_size, hidden_size]
        # logits: [batch, vocab/tp_size]
        logits = F.linear(x, self.weight)

        # ===== 张量并行：收集所有 GPU 的 logits =====
        if self.tp_size > 1:
            if self.tp_rank == 0:
                # 准备接收数据的 buffer
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None

            # gather: 从所有 GPU 收集数据到 rank 0
            dist.gather(logits, all_logits, 0)

            if self.tp_rank == 0:
                # 拼接所有 GPU 的 logits
                # all_logits: [tp_size, batch, vocab/tp_size]
                # cat 后: [batch, vocab]
                logits = torch.cat(all_logits, -1)
            else:
                logits = None

        return logits