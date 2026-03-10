"""
Attention 层实现

核心功能：
1. 使用 Triton 编写 KVCache 写入 kernel
2. 调用 FlashAttention 进行高效 Attention 计算
3. 支持 Prefill 和 Decode 两种模式
4. 支持前缀缓存
"""
import torch
from torch import nn
import triton
import triton.language as tl

# FlashAttention 库：高效的 Attention 实现（CUDA/C++）
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# ==================== Triton Kernel: KVCache 写入 ====================
@triton.jit
def store_kvcache_kernel(
    key_ptr,          # K 矩阵指针
    key_stride,       # K 步长
    value_ptr,       # V 矩阵指针
    value_stride,    # V 步长
    k_cache_ptr,     # KVCache K 指针
    v_cache_ptr,     # KVCache V 指针
    slot_mapping_ptr, # slot 映射指针
    D: tl.constexpr, # 维度大小（head_dim * num_kv_heads）
):
    """
    Triton kernel: 将 K/V 写入 KVCache

    原理：
    - 每个 thread 负责处理一个 token 的 K/V
    - slot_mapping 告诉每个 token 应该写入哪个位置

    对应 CUDA C++ 代码类似：
    __global__ void store_kvcache_kernel(...) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int slot = slot_mapping[idx];
        if (slot == -1) return;
        // 写入 K
        for (int d = 0; d < D; d++) {
            k_cache[slot * D + d] = key[idx * D + d];
        }
        // 写入 V
        for (int d = 0; d < D; d++) {
            v_cache[slot * D + d] = value[idx * D + d];
        }
    }
    """
    # 获取当前 thread处理的 token 索引
    idx = tl.program_id(0)
    # 读取该 token 对应的 slot 位置
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # -1 表示不需要写入

    # 计算 K/V 的偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 读取 K/V
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算写入 KVCache 的位置
    cache_offsets = slot * D + tl.arange(0, D)

    # 写入 KVCache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,       # [N, num_kv_heads, head_dim] 新计算的 K
    value: torch.Tensor,     # [N, num_kv_heads, head_dim] 新计算的 V
    k_cache: torch.Tensor,   # KVCache K
    v_cache: torch.Tensor,   # KVCache V
    slot_mapping: torch.Tensor  # [N] 每个 token 写入的位置
):
    """
    调用 Triton kernel 将 K/V 写入 KVCache

    Args:
        key: 新计算的 K，shape [N, num_kv_heads, head_dim]
        value: 新计算的 V，shape [N, num_kv_heads, head_dim]
        k_cache: KVCache K
        v_cache: KVCache V
        slot_mapping: 每个 token 应该写入的位置

    对应 C++:
    void store_kvcache(torch::Tensor key, torch::Tensor value,
                       torch::Tensor k_cache, torch::Tensor v_cache,
                       torch::Tensor slot_mapping) {
        int N = key.size(0);
        int D = key.size(1) * key.size(2);
        // 启动 Triton kernel
        store_kvcache_kernel<<<N, 1>>>(...);
    }
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 内存布局检查
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后维度连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 中间维度连续
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # KVCache 布局
    assert slot_mapping.numel() == N  # slot 数量匹配

    # 启动 Triton kernel，N 个并行任务
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D
    )


# ==================== Attention 模块 ====================
class Attention(nn.Module):
    """
    Attention 模块

    核心逻辑：
    1. 将新计算的 K/V 写入 KVCache
    2. 调用 FlashAttention 计算 Attention
    3. Prefill 和 Decode 使用不同的 FlashAttention 接口
    """

    def __init__(
        self,
        num_heads: int,        # Q 的头数
        head_dim: int,        # 每个头的维度
        scale: float,         # 缩放因子 (1/sqrt(d))
        num_kv_heads: int,    # K/V 的头数（可能少于 Q）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # KVCache 在 ModelRunner 中分配，这里只是占位
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,  # [total_seqlen, num_heads, head_dim] Q
        k: torch.Tensor,  # [total_seqlen, num_kv_heads, head_dim] K
        v: torch.Tensor,  # [total_seqlen, num_kv_heads, head_dim] V
    ) -> torch.Tensor:
        """
        Attention 前向传播

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            o: Attention 输出
        """
        # 获取上下文信息（Prefill/Decode、slot_mapping 等）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ===== 1. 写入 KVCache =====
        if k_cache.numel() and v_cache.numel():
            # 调用 Triton kernel 将 K/V 写入预分配的 KVCache
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # ===== 2. 计算 Attention =====
        if context.is_prefill:
            # Prefill阶段：处理完整序列
            if context.block_tables is not None:
                # 有前缀缓存：从 KVCache 读取
                k, v = k_cache, v_cache

            # 调用 FlashAttention (Variable Length)
            # 这是一个高度优化的 CUDA kernel
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,  # 下三角 Mask
                block_table=context.block_tables  # 前缀缓存块表
            )
        else:
            # Decode 阶段：只处理最后一个 token
            # 从 KVCache 读取历史 K/V
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # [bs, 1, num_heads, head_dim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个序列的上下文长度
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )

        return o
