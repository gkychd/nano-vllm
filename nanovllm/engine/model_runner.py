import pickle  # 序列化：用于多进程通信
import torch  # PyTorch
import torch.distributed as dist  # 分布式通信 (NCCL)
from multiprocessing.synchronize import Event  # 多进程事件
from multiprocessing.shared_memory import SharedMemory  # 共享内存

from nanovllm.config import Config  # 配置
from nanovllm.engine.sequence import Sequence  # 序列
from nanovllm.models.qwen3 import Qwen3ForCausalLM  # Qwen3 模型
from nanovllm.layers.sampler import Sampler  # 采样器
from nanovllm.utils.context import set_context, get_context, reset_context  # 上下文管理
from nanovllm.utils.loader import load_model  # 模型加载


class ModelRunner:
    """模型运行器：负责模型加载、推理执行、CUDA Graph 捕获、多进程通信"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器

        Args:
            config: 配置对象
            rank: 当前进程的 rank(0 为主进程)
            event: 多进程同步事件
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KVCache 块大小 = 256
        self.enforce_eager = config.enforce_eager  # 是否禁用 CUDA Graph
        self.world_size = config.tensor_parallel_size  # GPU 数量
        self.rank = rank  # 当前进程 rank
        self.event = event  # 多进程事件

        # ============ 1. 初始化分布式环境 (NCCL) ============
        dist.init_process_group("nccl", "tcp://localhost:23333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 设置当前 GPU

        # ============ 2. 加载模型 ============
        # 保存当前的 dtype 和 device 设置
        default_dtype = torch.get_default_dtype()
        # 设置模型使用 fp16/bf16
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")  # 默认创建 CUDA 张量
        # 创建 Qwen3 模型
        self.model = Qwen3ForCausalLM(hf_config)
        # 从 HuggingFace 加载权重
        load_model(self.model, config.model)
        # 创建采样器
        self.sampler = Sampler()

        # ============ 3. 预热和分配 KVCache ============
        self.warmup_model()  # 预热模型
        self.allocate_kv_cache()  # 预分配 KVCache 显存
        # 如果不禁用 CUDA Graph，则捕获计算图
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ============ 4. 多进程通信设置 (张量并行) ============
        if self.world_size > 1:
            if rank == 0:
                # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB
                dist.barrier()  # 同步
            else:
                # Worker 进程：连接共享内存
                dist.barrier()  # 同步
                self.shm = SharedMemory(name="nanovllm")
                # 进入循环，等待主进程指令
                self.loop()

    def exit(self):
        """退出清理：关闭共享内存、销毁 CUDA Graph、销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放 CUDA Graph
        torch.cuda.synchronize()
        dist.destroy_process_group()  # 销毁 NCCL 进程组

    def loop(self):
        """
        Worker 进程的主循环：等待主进程指令并执行
        类似于 RPC 服务器
        """
        while True:
            method_name, args = self.read_shm()  # 从共享内存读取指令
            self.call(method_name, *args)  # 执行方法
            if method_name == "exit":
                break

    def read_shm(self):
        """
        Worker 进程从共享内存读取指令
        格式: [4字节长度][序列化数据]
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待主进程通知
        # 读取前 4 字节获取数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 读取序列化数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 清除事件
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        主进程向共享内存写入指令
        格式: [4字节长度][序列化数据]
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入长度和数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有 Worker 进程
        for event in self.event:
            event.set()

    def call(self, method_name: str, *args):
        """
        调用指定方法（支持多进程）
        主进程会先写入共享内存，Worker 进程会读取并执行
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 主进程写入指令
        method = getattr(self, method_name, None)  # 获取方法
        return method(*args)  # 执行方法

    def warmup_model(self):
        """模型预热：用最大长度序列跑一次推理，触发 CUDA 编译 JIT"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KVCache 显存

        计算公式:
        num_blocks = (总显存 * 使用率 - 已用显存 - 峰值显存 + 当前显存) / 单块大小

        单块大小 = 2(K和V) * 层数 * 块大小 * KV头数 * 头维度 * dtype字节数

        分配后，将每个 Attention 层的 k_cache 和 v_cache 指向预分配的显存
        """
        config = self.config
        hf_config = config.hf_config
        # 获取显存信息
        free, total = torch.cuda.mem_get_info()  # 空闲显存, 总显存
        used = total - free  # 已用显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 峰值显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前显存
        # 计算每个 GPU 的 KV 头数（张量并行时需要分割）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算单个块的大小（字节）
        # 2 = K和V, num_hidden_layers = 层数, block_size = 256
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 计算可分配的块数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 分配 KVCache 显存: [2=K或V, 层数, 块数, 块大小, KV头数, 头维度]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim)

        # 将 KVCache 绑定到每个 Attention 层的 k_cache 和 v_cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备 block_tables（块表）

        用于告诉 FlashAttention 每个序列的 token 存储在哪些 KVCache 块中

        Args:
            seqs: 序列列表

        Returns:
            block_tables: [num_seqs, max_blocks] 的 int32 tensor
        """
        # 找到最大的块表长度
        max_len = max(len(seq.block_table) for seq in seqs)
        # 填充到相同长度（不够的用 -1 填充）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转为 tensor，使用 pinned memory 加速 CPU->GPU 传输
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段所需的输入数据

        Prefill 特点：一次性处理完整 prompt，可能包含前缀缓存

        返回:
            input_ids: 所有需要计算的 token
            positions: 对应的位置编码
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段所需的输入数据

        Decode 特点：每个序列只处理最后一个新 token

        返回:
            input_ids: 每个序列的最后一个 token
            positions: 对应的位置编码
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # Decode 只需要最后一个 token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)  # 位置 = 序列长度 - 1
            context_lens.append(len(seq))  # 上下文长度
            # slot_mapping: 告诉 KVCache 写入到哪个位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        # 转为 tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # 设置上下文（is_prefill=False 表示 decode 模式）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数

        只有 rank 0 需要准备采样参数（负责采样）

        Returns:
            temperatures: 每个序列的温度参数
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()  # 推理模式：禁用梯度计算
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型推理

        Args:
            input_ids: 输入 token IDs
            positions: 位置编码
            is_prefill: 是否是 prefill 阶段

        Returns:
            logits: 模型输出的 logits
        """
        # 条件判断：
        # 1. prefill 阶段：不用 CUDA Graph
        # 2. enforce_eager 模式：不用 CUDA Graph
        # 3. batch size > 512：不用 CUDA Graph（避免内存问题）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 普通执行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 加速（decode 阶段）
            bs = input_ids.size(0)
            context = get_context()
            # 根据 batch size 选择对应的 graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 更新 graph 的输入（复制数据到预分配的 buffer）
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)  # 先填充 -1
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放 graph（比重新执行快很多）
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        核心推理方法

        流程:
        1. 准备输入数据 (prepare_prefill / prepare_decode)
        2. 准备采样参数 (prepare_sample)
        3. 执行模型 (run_model)
        4. 采样下一个 token (sampler)

        Returns:
            token_ids: 采样到的 token ID 列表
        """
        # 1. 准备输入数据
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 2. 准备采样参数（只有 rank 0 需要）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 3. 执行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        # 4. 采样（只有 rank 0 需要）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 5. 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graph

        CUDA Graph 原理：
        - 第一次执行时，PyTorch 会记录整个计算图
        - 之后的执行只需要重放（replay）这个图，避免了每次重新编译的开销
        - 显著加速 decode 阶段

        为不同的 batch size 捕获不同的 graph:
        graph_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大 batch size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配输入输出的 buffer（避免每次分配）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 为不同的 batch size 捕获 graph
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # [1, 2, 4, 8, 16, 32, ...]
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获（有利于共享内存池）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # Warmup：先执行一次，确保所有 CUDA kernel 都编译完成
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # Capture：捕获计算图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 保存 graph pool（用于共享内存池）
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存预分配的 buffer，供运行时更新数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
