from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # 创建 KVCache 块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 两个核心队列：
        self.waiting: deque[Sequence] = deque() # 等待队列（WAITING 状态）
        self.running: deque[Sequence] = deque() # 运行队列（RUNNING 状态）

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # ======== 第一阶段：Prefill ========
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 从等待队列中选择序列进行 prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查是否超过限制
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            # 分配 KVCache 块
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 更新状态：WAITING → RUNNING
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # ======== 第二阶段：Decode ========
        # （如果上面没有 prefill 任务，才进入 decode）
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否可以追加（是否有足够的 KVCache 块）
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占：牺牲最后一个序列
                    self.preempt(self.running.pop())
                else:
                    # 没有其他序列可以抢占，只能抢占自己
                    self.preempt(seq)
                    break
            else:
                # 可以追加，分配新块
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) # 放回队列
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        #  """抢占：释放序列的 KVCache，放回等待队列"""
        # 抢占场景： 当 KVCache 不足时，把正在运行的序列踢回等待队列，释放其占用的 KVCache。
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        # """在模型推理后更新序列状态"""
        for seq, token_id in zip(seqs, token_ids):
            # 1. 追加新生成的 token
            seq.append_token(token_id)
            # 2. 检查是否完成 
            # 遇到 EOS
            # 达到最大长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq) # 释放 KVCache
                self.running.remove(seq)           # 从运行队列移除
