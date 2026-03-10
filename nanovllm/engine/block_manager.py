from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id # 块 ID
        self.ref_count = 0       # 引用计数（用于共享块）
        self.hash = -1           # 块的哈希值（前缀缓存用）
        self.token_ids = []      # 存储的 token（用于哈希比对）

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希和 token"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块（分配时调用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict() # 哈希 → 块ID 映射
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲块队列
        self.used_block_ids: set[int] = set() # 已使用块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算 token 序列的哈希值（用于前缀缓存）
            prefix: 前一个块的哈希值，用于计算级联哈希
        """
        h = xxhash.xxh64()   # 创建 xxhash64 哈希器
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))   # 合并前一个块的哈希
        h.update(np.array(token_ids).tobytes())      # 添加当前块的 token
        return h.intdigest()                         # 返回 64 位哈希

    def _allocate_block(self, block_id: int) -> Block:
        """分配指定 ID 的块"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)   # 从空闲队列移除
        self.used_block_ids.add(block_id)      # 加入已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放块（引用计数为 0 时）"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配 KVCache 块（前缀缓存核心！）"""
        assert not seq.block_table
        h = -1 # 初始哈希
        cache_miss = False # 是否缓存未命中
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)   # 获取第 i 块的 token
            # 计算哈希（级联：前一块哈希 + 当前块 token）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 查找是否有匹配的缓存块
            block_id = self.hash_to_block_id.get(h, -1)
            # 检查缓存是否有效
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 缓存未命中：从空闲块分配新的
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中！
                seq.num_cached_tokens += self.block_size  # 增加缓存计数
                if block_id in self.used_block_ids:
                    # 块已被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块空闲，分配使用
                    block = self._allocate_block(block_id)
            # 更新哈希映射
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加新 token(是否需要新块)"""
        # 如果是块的边界位置（len % 256 == 1），需要新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Decode 阶段追加 token 时调用"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 刚好是块边界，需要分配新块
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 块满了，更新哈希（用于下次前缀缓存）
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 块未满，不需要操作
            assert last_block.hash == -1
