from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()  # 等待调度
    RUNNING = auto()  # 正在推理
    FINISHED = auto() # 完成


class Sequence:
    block_size = 256   # 类变量：KVCache 块大小（所有实例共享）
    counter = count()  # 类变量：全局计数器，用于生成唯一 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    #1.属性方法
    # 获取序列长度
    def __len__(self):
        return self.num_tokens

    # 下标访问
    def __getitem__(self, key):
        return self.token_ids[key]

    # 是否完成
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 已生成的 token 数（不含 prompt）
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # prompt token 列表
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 生成的 token 列表
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    #2. 块管理相关
    # 已缓存的块数
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # 总块数（向上取整）
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个块的 token 数
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取第 i 块的 token
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    #3. 追加token
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    #4. 序列化（多进程通信用） 把结构体打包成字节流
    # 序列化：把对象变成可传输的字节流
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    # 反序列化   把字节流解析回结构体
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
