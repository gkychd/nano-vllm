"""
上下文管理模块

作用：存储 Prefill/Decode 阶段的上下文信息，供各层使用

为什么需要全局上下文？
- Attention 层需要知道当前是 Prefill 还是 Decode
- 需要知道序列长度信息（cu_seqlens）
- 需要知道 KVCache 的 slot_mapping
- 需要知道 block_tables（前缀缓存）

实现方式：
- 使用全局变量 _CONTEXT 存储上下文
- 类似线程局部存储（Thread Local Storage），但这里是进程级别
- 在 ModelRunner 的 prepare_prefill / prepare_decode 中设置
- 在 run 结束后 reset
"""
from dataclasses import dataclass  # 数据类装饰器
import torch


@dataclass
class Context:
    """
    上下文数据类

    属性说明：
    - is_prefill: 是否是 Prefill 阶段
    - cu_seqlens_q: Query 的累计序列长度（用于变长序列）
    - cu_seqlens_k: Key/Value 的累计序列长度
    - max_seqlen_q: Query 的最大序列长度
    - max_seqlen_k: Key/Value 的最大序列长度
    - slot_mapping: KVCache 槽位映射
    - context_lens: 每个序列的上下文长度（Decode 阶段）
    - block_tables: KVCache 块表（前缀缓存）

    对应 C++ 结构体：
    struct Context {
        bool is_prefill;
        int* cu_seqlens_q;
        int* cu_seqlens_k;
        int max_seqlen_q;
        int max_seqlen_k;
        int* slot_mapping;
        int* context_lens;
        int* block_tables;
    };
    """
    is_prefill: bool = False                    # Prefill 或 Decode
    cu_seqlens_q: torch.Tensor | None = None   # [num_seqs + 1]，累计长度
    cu_seqlens_k: torch.Tensor | None = None   # [num_seqs + 1]
    max_seqlen_q: int = 0                       # 最大 query 长度
    max_seqlen_k: int = 0                       # 最大 key 长度
    slot_mapping: torch.Tensor | None = None   # [batch] 每个 token 的槽位
    context_lens: torch.Tensor | None = None    # [batch] 每个序列的上下文长度
    block_tables: torch.Tensor | None = None    # [batch, max_blocks]


# 全局上下文变量
_CONTEXT = Context()


def get_context():
    """
    获取当前上下文

    Returns:
        Context: 全局上下文对象

    使用示例（在 Attention 中）：
    context = get_context()
    if context.is_prefill:
        # Prefill 逻辑
    else:
        # Decode 逻辑
    """
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None
):
    """
    设置全局上下文

    在 ModelRunner.prepare_prefill / prepare_decode 中调用

    Args:
        is_prefill: 是否是 Prefill 阶段
        cu_seqlens_q: Query 累计序列长度
        cu_seqlens_k: Key 累计序列长度
        max_seqlen_q: Query 最大长度
        max_seqlen_k: Key 最大长度
        slot_mapping: KVCache 槽位映射
        context_lens: 上下文长度
        block_tables: 块表
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables
    )


def reset_context():
    """
    重置全局上下文

    在 run 结束后调用，清空上下文信息
    """
    global _CONTEXT
    _CONTEXT = Context()
