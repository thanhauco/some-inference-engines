"""PagedAttention, the memory idea behind vLLM.

During decoding every token's attention keys and values (the KV cache) must stay in GPU
memory. Older servers reserved one *contiguous* region per request sized for the maximum
sequence length, so a 200-token answer in a 4096-token slot wasted ~95% of it, and that
waste - not compute - capped how many requests fit in a batch.

PagedAttention borrows virtual memory's trick: split the KV cache into fixed-size
**blocks** (pages) and give each sequence a **block table** mapping logical token
positions to physical blocks anywhere in memory. A sequence grows one block at a time,
so the only waste is the unfilled tail of its last block. Blocks are reference counted,
so parallel samples or beams that share a prompt share its blocks (copy-on-write).

This module is a faithful, framework-free model of that block manager.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class OutOfBlocks(RuntimeError):
    """No free KV block: the scheduler must wait, or preempt a running sequence."""


class BlockAllocator:
    """Physical KV blocks with reference counts."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))
        self.ref_counts = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks("KV cache is full")
        block = self._free.pop()
        self.ref_counts[block] = 1
        return block

    def share(self, block: int) -> int:
        self.ref_counts[block] += 1
        return block

    def release(self, block: int) -> None:
        if self.ref_counts[block] <= 0:
            raise ValueError(f"block {block} is already free")
        self.ref_counts[block] -= 1
        if self.ref_counts[block] == 0:
            self._free.append(block)


@dataclass
class CacheStats:
    sequences: int
    tokens: int
    used_blocks: int
    free_blocks: int
    slot_utilization: float
    """Tokens stored / token slots in allocated blocks (1.0 = no internal fragmentation)."""


class PagedKVCache:
    """Block tables for many sequences on top of one :class:`BlockAllocator`."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self.copies = 0
        """Copy-on-write block copies performed (a real engine would memcpy a KV block)."""

    # -- capacity -----------------------------------------------------------------------
    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.allocator.num_free

    def needs_new_block(self, seq_id: int) -> bool:
        """Would appending one token to ``seq_id`` require a fresh physical block?"""
        length = self.seq_lens[seq_id]
        if length % self.block_size == 0:
            return True
        last = self.block_tables[seq_id][-1]
        return self.allocator.ref_counts[last] > 1  # shared tail -> copy-on-write

    # -- lifecycle ----------------------------------------------------------------------
    def add_sequence(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a prompt of ``num_tokens`` (the prefill)."""
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        if not self.can_allocate(num_tokens):
            raise OutOfBlocks(f"need {self.blocks_needed(num_tokens)} blocks")
        table = [self.allocator.allocate() for _ in range(self.blocks_needed(num_tokens))]
        self.block_tables[seq_id] = table
        self.seq_lens[seq_id] = num_tokens
        return table

    def append_token(self, seq_id: int) -> None:
        """Store the KV of one newly decoded token."""
        table = self.block_tables[seq_id]
        length = self.seq_lens[seq_id]
        if length % self.block_size == 0:
            table.append(self.allocator.allocate())
        else:
            last = table[-1]
            if self.allocator.ref_counts[last] > 1:
                # The tail block is shared with a fork: give this sequence a private copy.
                table[-1] = self.allocator.allocate()
                self.allocator.release(last)
                self.copies += 1
        self.seq_lens[seq_id] = length + 1

    def fork(self, parent: int, child: int) -> None:
        """Start ``child`` as a copy of ``parent`` without copying any KV (parallel sampling)."""
        if child in self.block_tables:
            raise ValueError(f"sequence {child} already exists")
        self.block_tables[child] = [self.allocator.share(b) for b in self.block_tables[parent]]
        self.seq_lens[child] = self.seq_lens[parent]

    def free(self, seq_id: int) -> None:
        for block in self.block_tables.pop(seq_id):
            self.allocator.release(block)
        del self.seq_lens[seq_id]

    # -- introspection ------------------------------------------------------------------
    def stats(self) -> CacheStats:
        used = self.allocator.num_blocks - self.allocator.num_free
        tokens = sum(self.seq_lens.values())
        unique_slots = used * self.block_size
        # Shared blocks hold the same tokens for several sequences; count them once.
        unique_tokens = 0
        seen: set[int] = set()
        for seq_id, table in self.block_tables.items():
            length = self.seq_lens[seq_id]
            for i, block in enumerate(table):
                if block not in seen:
                    seen.add(block)
                    unique_tokens += min(self.block_size, length - i * self.block_size)
        return CacheStats(
            sequences=len(self.block_tables),
            tokens=tokens,
            used_blocks=used,
            free_blocks=self.allocator.num_free,
            slot_utilization=unique_tokens / unique_slots if unique_slots else 1.0,
        )


def contiguous_utilization(seq_lens: Iterable[int], max_seq_len: int) -> float:
    """Slot utilization if every sequence pre-reserves ``max_seq_len`` contiguous slots."""
    lens = list(seq_lens)
    return sum(lens) / (len(lens) * max_seq_len) if lens else 1.0


def paged_utilization(seq_lens: Iterable[int], block_size: int) -> float:
    """Slot utilization with paging: only the last block of each sequence is partly empty."""
    lens = list(seq_lens)
    slots = sum(-(-n // block_size) * block_size for n in lens)
    return sum(lens) / slots if slots else 1.0
