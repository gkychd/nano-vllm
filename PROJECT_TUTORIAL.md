# Nano-VLLM 项目详解

> 面向 C/C++ 工程师的 Python vLLM 实现学习指南

## 目录

1. [项目概述](#项目概述)
2. [整体架构](#整体架构)
3. [核心模块详解](#核心模块详解)
4. [数据流与执行流程](#数据流与执行流程)
5. [关键技术点](#关键技术点)
6. [代码文件索引](#代码文件索引)

---

## 项目概述

**Nano-VLLM** 是一个轻量级的 vLLM 实现，专注于 LLM（大型语言模型）推理加速。

### 与 vLLM 的关系

| 特性 | vLLM | Nano-VLLM |
|------|------|-----------|
| 语言 | C++/Python 混合 | Python 为主 |
| 核心计算 | CUDA C++ | FlashAttention (CUDA) |
| KVCache | C++ 自研 | Triton + FlashAttention |
| 张量并行 | NCCL C++ | PyTorch Distributed |
| 代码量 | 大 | 小（可学习） |

### 核心技术栈

- **PyTorch**: 深度学习框架
- **Triton**: GPU Kernel 编写（Python 风格）
- **FlashAttention**: 高效 Attention 实现
- **NCCL**: NVIDIA 多 GPU 通信库
- **xxhash**: 前缀缓存哈希计算

---

## 整体架构

### 模块关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        example.py                               │
│                    (用户入口，LLM 类)                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      llm_engine.py                              │
│                  (核心引擎，调度协调)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Scheduler   │  │BlockManager │  │     ModelRunner         │  │
│  │ (调度器)     │  │(块管理器)    │  │    (模型运行器)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     model_runner.py                             │
│                (模型执行，CUDA Graph)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Prefill     │  │  Decode     │  │   CUDA Graph            │  │
│  │ (预填充)     │  │  (解码)      │  │   (图捕获与重放)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     qwen3.py (模型)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │Embedding    │  │ DecoderLayer│  │   LM Head               │  │
│  │(词嵌入)      │  │  × 28层     │  │   (输出层)               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│   attention.py  │ │   linear.py     │ │  rotary_embedding.py    │
│  (Attention)    │ │  (Linear 层)    │ │   (RoPE 位置编码)         │
│ ┌─────────────┐ │  ┌─────────────┐  │ └─────────────────────────┘
│ │Triton Kernel│ │  │ColumnParallel│ │
│ │(KVCache写入) │ │  │RowParallel  │  │ ┌─────────────────────────┐
│ └─────────────┘ │  └─────────────┘  │ │  layernorm.py           │
└─────────────────┘                   │ │  (RMSNorm)              │
                                      │ └─────────────────────────┘
                                      │ ┌─────────────────────────┐
                                      │ │  activation.py          │
                                      │ │  (SiluAndMul)           │
                                      │ └─────────────────────────┘
                                      │ ┌─────────────────────────┐
                                      │ │  sampler.py             │
                                      │ │  (采样器)                │
                                      │ └─────────────────────────┘
                                      │ ┌─────────────────────────┐
                                      │ │  embed_head.py          │
                                      │ │  (Embedding/LMHead)     │
                                      │ └─────────────────────────┘
                                      └─────────────────────────────┘
```

### 各模块职责

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| `llm_engine.py` | 整体协调，API 入口 | `LLMEngine.generate()` |
| `scheduler.py` | Prefill/Decode 调度 | `Scheduler.schedule()` |
| `block_manager.py` | KVCache 块分配，前缀缓存 | `BlockManager.allocate()` |
| `sequence.py` | 序列状态管理 | `Sequence` |
| `model_runner.py` | 模型执行，CUDA Graph | `ModelRunner.run()` |
| `attention.py` | KVCache 写入，Attention 计算 | `store_kvcache()`, `Attention.forward()` |
| `linear.py` | 张量并行 Linear 层 | `ColumnParallelLinear`, `RowParallelLinear` |
| `rotary_embedding.py` | RoPE 旋转位置编码 | `RotaryEmbedding` |
| `sampler.py` | Token 采样 | `Sampler.forward()` |

---

## 核心模块详解

### 1. 推理流程 (Prefill + Decode)

LLM 推理分为两个阶段：

#### Prefill 阶段（预填充）
- **一次性**处理完整 prompt
- **特点**：计算密集，但只执行一次
- **输出**：KVCache + 第一个新 token

#### Decode 阶段（解码）
- **逐个**生成新 token
- **特点**：内存密集（需要读取完整 KVCache）
- **循环执行**直到生成 EOS 或达到最大长度

```
输入: "今天天气"
  │
  ▼
┌──────────────────────┐
│     Prefill          │
│  处理完整 prompt      │
│  输出: 第一个新 token  │
└──────────────────────┘
  │ "很好"
  ▼
┌──────────────────────┐
│     Decode           │
│  生成 "很好"          │
└──────────────────────┘
  │ "，"
  ▼
┌──────────────────────┐
│     Decode           │
│  生成 "，"            │
└──────────────────────┘
  │ ...
  ▼
输出: "今天天气很好，适合出去玩"
```

### 2. KVCache 管理

#### Block Manager

```python
# 块分配流程
block_manager.allocate(seq)  # 为序列分配 KVCache 块

# 前缀缓存匹配
1. 计算每个块的 hash（xxhash64）
2. 通过 hash 查找匹配的缓存块
3. 命中则复用，未命中则分配新块
```

#### KVCache 写入 (Triton)

```python
# Triton Kernel: 并行写入 K/V 到显存
@triton.jit
def store_kvcache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, slot_mapping):
    idx = tl.program_id(0)           # thread ID
    slot = tl.load(slot_mapping_ptr + idx)  # 槽位
    if slot == -1: return            # 无效槽位
    # 写入 K/V 到 KVCache
    tl.store(k_cache_ptr + slot, key)
    tl.store(v_cache_ptr + slot, value)
```

### 3. 调度器 (Scheduler)

```python
def schedule(self):
    # 第一阶段：Prefill
    while waiting and can_allocate:
        seq = waiting.pop()
        block_manager.allocate(seq)  # 分配块
        prefill(seq)

    # 第二阶段：Decode（如果没有 prefill 任务）
    while running:
        seq = running.pop()
        block_manager.may_append(seq)  # 追加新块
        decode(seq)
```

### 4. 张量并行 (Tensor Parallelism)

#### 概念
- 将模型权重分割到多个 GPU
- 每个 GPU 计算部分结果
- 通过通信原语（all_reduce）汇总

#### Linear 层分割

**Column Parallel (列并行)**:
```
权重: W [4096, 4096] → W0 [2048, 4096] (GPU 0), W1 [2048, 4096] (GPU 1)
输入: x [batch, 4096]
GPU 0: y0 = x @ W0.T [batch, 2048]
GPU 1: y1 = x @ W1.T [batch, 2048]
输出: [y0, y1] (拼接)
```

**Row Parallel (行并行)**:
```
权重: W [4096, 4096] → W0 [4096, 2048] (GPU 0), W1 [4096, 2048] (GPU 1)
输入: x [batch, 4096] → x0 [batch, 2048] (GPU 0), x1 [batch, 2048] (GPU 1)
GPU 0: y0 = x0 @ W0.T
GPU 1: y1 = x1 @ W1.T
输出: y = y0 + y1 (all_reduce)
```

### 5. CUDA Graph 加速

```python
# 捕获阶段
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outputs = model(input_ids, positions)  # 记录计算图

# 运行阶段（只需重放，无需重新编译）
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph.replay()  # 快速执行
return graph_vars["outputs"][:bs]
```

### 6. RoPE 旋转位置编码

```python
# 核心公式
def apply_rotary_emb(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin  # 旋转矩阵第一行
    y2 = x2 * cos + x1 * sin  # 旋转矩阵第二行
    return torch.cat((y1, y2), dim=-1)
```

数学原理：
```
[x']   [cos(θ)  -sin(θ)]   [x]
[  ] = [               ] * [ ]
[y']   [sin(θ)   cos(θ)]   [y]
```

### 7. 采样器 (Gumbel-Max)

```python
# Gumbel-Max 采样：高效的离散采样
def forward(self, logits, temperatures):
    # 1. 温度缩放
    logits = logits / temperature

    # 2. Softmax
    probs = softmax(logits)

    # 3. Gumbel-Max 采样
    noise = exponential(1)  # Gumbel 噪声
    scores = probs / noise
    return argmax(scores)
```

---

## 数据流与执行流程

### 完整推理流程

```
用户请求: "你好" → max_tokens=100

  │
  ▼
LLMEngine.generate()
  │
  ├── 1. 创建 Sequence
  │   sequence = Sequence([token_ids], sampling_params)
  │
  ├── 2. Scheduler.schedule()
  │   ├── 分配 KVCache 块
  │   └── 返回 scheduled_seqs, is_prefill
  │
  ├── 3. ModelRunner.run()
  │   │
  │   ├── prepare_prefill() / prepare_decode()
  │   │   ├── input_ids, positions
  │   │   ├── cu_seqlens_q/k
  │   │   ├── slot_mapping
  │   │   └── block_tables
  │   │
  │   ├── run_model()
  │   │   ├── model(input_ids, positions)  # 前向传播
  │   │   │   ├── embed_tokens(input_ids)
  │   │   │   ├── DecoderLayer × N
  │   │   │   │   ├── attention(q, k, v)
  │   │   │   │   │   ├── store_kvcache()  # 写入 KVCache
  │   │   │   │   │   └── flash_attn()     # Attention 计算
  │   │   │   │   └── mlp(hidden_states)
  │   │   │   └── lm_head(hidden_states)
  │   │   │
  │   │   └── compute_logits(hidden_states)
  │   │
  │   └── sampler(logits)  # 采样下一个 token
  │
  ├── 4. 更新 Sequence
  │   ├── append_token(token_id)
  │   └── 检查是否结束
  │
  └── 循环直到完成
```

### Prefill vs Decode 数据流

#### Prefill 阶段
```
输入: [BOS, "今", "天", "天", "气", "很", "好"]

ModelRunner.prepare_prefill():
  input_ids = [所有 token]
  cu_seqlens_q = [0, 8]
  cu_seqlens_k = [0, 8]
  slot_mapping = [0, 1, 2, ..., 7]  # 写入位置
  is_prefill = True

FlashAttention 调用:
  flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q=8,
    cu_seqlens_q=[0, 8],
    max_seqlen_k=8,
    cu_seqlens_k=[0, 8]
  )
```

#### Decode 阶段
```
输入: 最后一个 token

ModelRunner.prepare_decode():
  input_ids = [last_token]  # 只有一个
  slot_mapping = [block_id * 256 + position]  # 写入位置
  context_lens = [total_len]  # 上下文长度
  is_prefill = False

FlashAttention 调用:
  flash_attn_with_kvcache(
    q,
    k_cache, v_cache,  # 从缓存读取
    cache_seqlens=[context_len],
    block_table=block_tables
  )
```

---

## 关键技术点

### 1. 为什么 Python 可以做到高效？

核心计算并不在 Python：

| 操作 | 实现方式 |
|------|----------|
| Attention 计算 | FlashAttention (CUDA C++) |
| Matrix Multiply | cuBLAS (CUDA) |
| KVCache 写入 | Triton Kernel (编译为 CUDA) |
| 分布式通信 | NCCL (CUDA C++) |
| 模型前向 | PyTorch (CUDA Kernel) |

Python 只负责：
- 控制流（调度、循环）
- 数据准备（整理 tensor）
- 调用底层库

### 2. Triton vs CUDA C++

**Triton**:
```python
@triton.jit
def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Python 风格的 CUDA 编程
```

**CUDA C++**:
```cuda
__global__ void kernel(float *x, float *y) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    // C++ 风格的 CUDA 编程
}
```

Triton 优势：
- Python 语法，更易上手
- 自动优化内存访问
- 无需编写 CUDA Boilerplate

### 3. 前缀缓存 (Prefix Caching)

```python
# 哈希计算
h = xxhash(token_ids)  # 64 位哈希

# 缓存查找
if h in hash_to_block_id:
    # 命中！复用已计算的 KVCache
    block_id = hash_to_block_id[h]
else:
    # 未命中，需要重新计算
```

### 4. 张量并行通信

```python
# all_reduce: 所有 GPU 求和
dist.all_reduce(y)

# gather: 收集到指定 GPU
dist.gather(logits, all_logits, dst=0)
```

---

## 代码文件索引

### 核心引擎

| 文件 | 行数 | 功能 |
|------|------|------|
| [llm_engine.py](nanovllm/engine/llm_engine.py) | ~100 | API 入口，整体协调 |
| [scheduler.py](nanovllm/engine/scheduler.py) | ~90 | Prefill/Decode 调度 |
| [block_manager.py](nanovllm/engine/block_manager.py) | ~137 | KVCache 块分配，前缀缓存 |
| [sequence.py](nanovllm/engine/sequence.py) | ~100 | 序列状态管理 |
| [model_runner.py](nanovllm/engine/model_runner.py) | ~424 | 模型执行，CUDA Graph |

### 模型层

| 文件 | 行数 | 功能 |
|------|------|------|
| [qwen3.py](nanovllm/models/qwen3.py) | ~215 | Qwen3 模型实现 |
| [attention.py](nanovllm/layers/attention.py) | ~205 | Attention + Triton KVCache |
| [linear.py](nanovllm/layers/linear.py) | ~154 | 张量并行 Linear |
| [rotary_embedding.py](nanovllm/layers/rotary_embedding.py) | ~62 | RoPE 位置编码 |
| [layernorm.py](nanovllm/layers/layernorm.py) | ~51 | RMSNorm |
| [activation.py](nanovllm/layers/activation.py) | ~15 | SiLU/SwiGLU |
| [sampler.py](nanovllm/layers/sampler.py) | ~65 | Token 采样 |
| [embed_head.py](nanovllm/layers/embed_head.py) | ~67 | Embedding + LM Head |

### 工具模块

| 文件 | 行数 | 功能 |
|------|------|------|
| [context.py](nanovllm/utils/context.py) | ~28 | 上下文管理 |
| [loader.py](nanovllm/utils/loader.py) | ~29 | 模型权重加载 |
| [config.py](nanovllm/config.py) | 配置 | 配置类 |
| [sampling_params.py](nanovllm/sampling_params.py) | 参数 | 采样参数 |

---

## 学习路径建议

### 入门顺序

1. **理解推理流程**
   - 阅读 `example.py`
   - 理解 Prefill vs Decode

2. **理解数据流**
   - 阅读 `llm_engine.py`
   - 理解 Scheduler 调度逻辑

3. **理解 KVCache**
   - 阅读 `block_manager.py`
   - 理解块分配和前缀缓存

4. **理解模型执行**
   - 阅读 `model_runner.py`
   - 理解 CUDA Graph 原理

5. **理解底层算子**
   - 阅读 `attention.py` (Triton + FlashAttention)
   - 阅读 `linear.py` (张量并行)
   - 阅读 `rotary_embedding.py` (位置编码)

### C++ 工程师特别关注

| Python 特性 | C++ 等价 |
|-------------|----------|
| `@property` | `getter` 方法 |
| `@torch.compile` | CUDA JIT 编译 |
| `torch.Tensor` | `torch::Tensor` |
| `nn.Module` | `torch::nn::Module` |
| `dist.all_reduce` | `ncclAllReduce` |
| `@triton.jit` | `__global__` |
| `deque` | `std::deque` |
| `dataclass` | 结构体 |

---

*文档生成时间: 2026-03-10*
