import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 多进程：创建 ModelRunner 进程（张量并行）
        self.ps = []
        self.events = []
        # 获取 spawn 模式的进程上下文
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): #默认为1
            event = ctx.Event() # 创建事件用于进程间通信
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程 (rank 0) 的 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id # 设置结束符 ID
        # 创建调度器
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果是字符串，先 tokenize (编码)
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 创建 Sequence 对象
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    #调用 scheduler 决定调度哪些序列
    #调用 model_runner 执行模型推理
    #调用 scheduler 更新序列状态
    #step() 是整个引擎的核心循环！
    def step(self):
        # 1. 调度：选择要处理的序列，返回 (seqs, is_prefill)
        seqs, is_prefill = self.scheduler.schedule()
        # 2. 模型运行：执行推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 3. 后处理：更新序列状态
        self.scheduler.postprocess(seqs, token_ids)
        # 4. 收集已完成的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 5. 计算处理的 token 数（正数=prefill，负数=decode）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 创建进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 扩展采样参数为列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 核心循环：直到所有序列完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 计算吞吐量
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 收集输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 整理输出顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码 token_ids 为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
