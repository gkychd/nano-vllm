"""
Qwen3 模型实现

模型架构（Transformer Decoder）：
┌─────────────────────────────────────────────┐
│                  Input                       │
│              [batch, seqlen]                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Embedding Layer                    │
│     VocabParallelEmbedding (TP)             │
│         [batch, seqlen, hidden]              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Decoder Layers × N                 │
│  ┌─────────────────────────────────────────┐ │
│  │ 1. RMSNorm (input_layernorm)           │ │
│  │ 2. Qwen3Attention                      │ │
│  │    - QKVParallelLinear (TP)            │ │
│  │    - RoPE                              │ │
│  │    - Attention                         │ │
│  │    - RowParallelLinear (TP)            │ │
│  │ 3. RMSNorm (post_attention_layernorm) │ │
│  │ 4. Qwen3MLP                          │ │
│  │    - MergedColumnParallelLinear (TP)  │ │
│  │    - SiluAndMul                       │ │
│  │    - RowParallelLinear (TP)           │ │
│  └─────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Output RMSNorm                     │
│              [batch, seqlen, hidden]         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│            LM Head (TP)                      │
│         ParallelLMHead                        │
│         [batch, seqlen, vocab]               │
└─────────────────────────────────────────────┘
"""
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 Attention 模块

    结构：
    - QKVParallelLinear: 一次计算 Q、K、V（列并行）
    - RMSNorm: Q、K 的归一化（Qwen3 特有）
    - RoPE: 旋转位置编码
    - Attention: FlashAttention 计算
    - RowParallelLinear: 输出投影（行并行）

    数据流：
    hidden_states → QKVLinear → 切分 Q,K,V → RMSNorm → RoPE → Attention → O_proj → output

    对应 C++ 类：
    class Qwen3Attention {
        QKVParallelLinear qkv_proj;
        RMSNorm q_norm, k_norm;
        RotaryEmbedding rotary_emb;
        Attention attn;
        RowParallelLinear o_proj;

        torch::Tensor forward(torch::Tensor positions,
                              torch::Tensor hidden_states) {
            // 1. QKV 投影
            auto qkv = qkv_proj(hidden_states);
            auto [q, k, v] = qkv.split({q_size, kv_size, kv_size});

            // 2. Reshape 和归一化
            q = q.view(-1, num_heads, head_dim);
            k = k.view(-1, num_kv_heads, head_dim);
            v = v.view(-1, num_kv_heads, head_dim);
            q = q_norm(q);
            k = k_norm(k);

            // 3. RoPE 位置编码
            auto [q, k] = rotary_emb(positions, q, k);

            // 4. Attention
            auto o = attn(q, k, v);

            // 5. 输出投影
            return o_proj(o.flatten(1, -1));
        }
    };
    """

    def __init__(
        self,
        hidden_size: int,              # 隐藏层维度（如 4096）
        num_heads: int,                 # Attention 头数（如 32）
        num_kv_heads: int,              # KV 头数（如 32，GQA）
        max_position: int = 4096 * 32,  # 最大位置
        head_dim: int | None = None,   # 头维度
        rms_norm_eps: float = 1e-06,    # RMSNorm epsilon
        qkv_bias: bool = False,         # QKV 是否有偏置
        rope_theta: float = 10000,      # RoPE 基础频率
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 张量并行大小

        # ===== 头数计算（张量并行分割） =====
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        # 每个 GPU 的头数
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # ===== 维度计算 =====
        # head_dim：每个头的维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # Q 的总维度
        self.q_size = self.num_heads * self.head_dim
        # KV 的总维度
        self.kv_size = self.num_kv_heads * self.head_dim
        # Attention 缩放因子
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # ===== 模块初始化 =====
        # QKV 投影：一次计算 Q、K、V（列并行）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # 输出投影（行并行，需要 all_reduce）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # RoPE 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        # Attention（FlashAttention）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # Qwen3 特有：Q 和 K 的 RMSNorm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,   # 位置编码
        hidden_states: torch.Tensor,  # 输入 hidden states
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            positions: 位置 tensor，[batch]
            hidden_states: 输入 hidden states，[batch, hidden_size]

        Returns:
            输出 hidden states，[batch, hidden_size]
        """
        # 1. QKV 投影
        qkv = self.qkv_proj(hidden_states)

        # 2. 沿最后一个维度切分 Q、K、V
        # qkv: [batch, q_size + kv_size + kv_size]
        # split 后：
        # q: [batch, q_size]
        # k: [batch, kv_size]
        # v: [batch, kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 3. Reshape 为 [batch, num_heads, head_dim] 格式
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 4. Qwen3 特有：Q、K 的 RMSNorm 归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 5. 应用 RoPE 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 6. FlashAttention 计算
        o = self.attn(q, k, v)

        # 7. 输出投影（行并行）
        # flatten(1, -1): [batch, num_heads, head_dim] → [batch, num_heads * head_dim]
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP 模块

    SwiGLU 结构（Qwen3 使用）：
    - gate_proj: 门控_proj
    - up_proj: 上_proj
    - act_fn: SiLU(gate) * up
    - down_proj: 下_proj

    公式：down_proj(silu(gate_proj(x)) * up_proj(x))

    结构：
    ┌─────────────┐     ┌─────────────┐
    │ gate_proj  │     │  up_proj   │
    │ (Column TP) │     │ (Column TP) │
    └──────┬──────┘     └──────┬──────┘
           │                  │
           ▼                  ▼
        ┌─────────────────────────────┐
        │      SiluAndMul            │
        │   output = silu(gate)*up   │
        └─────────────┬──────────────┘
                      │
                      ▼
           ┌─────────────────┐
           │   down_proj    │
           │  (Row Parallel)│
           └────────┬────────┘
                    │
                    ▼
           [batch, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,           # 输入维度
        intermediate_size: int,     # 中间维度（通常 2-4 倍 hidden_size）
        hidden_act: str,           # 激活函数
    ) -> None:
        super().__init__()
        # gate 和 up 合并为一个投影（列并行）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # 两个输出
            bias=False,
        )
        # 下投影（行并行）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        # Qwen3 使用 SiLU（Swish）激活函数
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # gate_up: [batch, 2 * intermediate_size]
        gate_up = self.gate_up_proj(x)
        # 应用 SiLU * up
        x = self.act_fn(gate_up)
        # 下投影
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Decoder 层

    结构（Pre-Norm）：
    1. input_layernorm → attention → residual
    2. post_attention_layernorm → mlp → residual

    Pre-Norm 优点：
    - 训练更稳定
    - 深层梯度不易消失
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # Attention 模块
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP 模块
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            positions: 位置编码
            hidden_states: 输入
            residual: 残差连接（如果是第一层则为 None）

        Returns:
            (output_hidden_states, output_residual)
        """
        # ===== 1. Attention 块 =====
        if residual is None:
            # 第一个 Decoder 层：先计算 norm，再保存残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：融合的 RMSNorm（输出 norm + residual）
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention
        hidden_states = self.self_attn(positions, hidden_states)

        # ===== 2. MLP 块 =====
        # 同样先 norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 基础模型

    结构：
    - Embedding → Decoder Layers × N → RMSNorm
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入层（张量并行）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # N 个 Decoder 层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 输出 RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        # 1. 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # 2. 逐层通过 Decoder
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 3. 最终 RMSNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型

    结构：
    - Qwen3Model（主体）
    - LM Head（输出层）

    作用：将 hidden states 转换为 vocab 大小的 logits
    """

    # 权重映射：用于从 HuggingFace 加载权重
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # 主体模型
        self.model = Qwen3Model(config)
        # LM Head（张量并行）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 权重共享（可选）
        if config.tie_word_embeddings:
            # LM Head 与 Embedding 共享权重
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """返回 hidden states（未经过 LM Head）"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 logits

        Args:
            hidden_states: [batch, seqlen, hidden_size]

        Returns:
            logits: [batch, seqlen, vocab_size]
        """
        return self.lm_head(hidden_states)
