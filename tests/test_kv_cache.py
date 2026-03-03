import pytest

from inference_engines.concepts.kv_cache import (
    OutOfBlocks,
    PagedKVCache,
    contiguous_utilization,
    paged_utilization,
)


def test_sequences_grow_one_block_at_a_time():
    cache = PagedKVCache(num_blocks=8, block_size=4)
    cache.add_sequence(0, num_tokens=5)  # 2 blocks
    assert len(cache.block_tables[0]) == 2
    for _ in range(3):  # fills block 2 up to 8 tokens
        cache.append_token(0)
    assert len(cache.block_tables[0]) == 2
    cache.append_token(0)  # token 9 needs a third block
    assert len(cache.block_tables[0]) == 3
    assert cache.stats().free_blocks == 5


def test_blocks_need_not_be_contiguous_and_are_recycled():
    cache = PagedKVCache(num_blocks=4, block_size=2)
    cache.add_sequence(1, 2)
    cache.add_sequence(2, 2)
    cache.add_sequence(3, 2)
    cache.free(2)
    cache.add_sequence(4, 4)  # reuses the hole left by seq 2 plus the last free block
    assert sorted(b for t in cache.block_tables.values() for b in t) == [0, 1, 2, 3]
    with pytest.raises(OutOfBlocks):
        cache.add_sequence(5, 1)


def test_fork_shares_blocks_and_copies_on_write():
    cache = PagedKVCache(num_blocks=8, block_size=4)
    cache.add_sequence(0, 6)  # blocks: [full, half]
    cache.fork(0, 1)
    assert cache.block_tables[0] == cache.block_tables[1]
    assert cache.stats().used_blocks == 2  # forking is free
    cache.append_token(1)  # shared half block -> private copy for seq 1
    assert cache.copies == 1
    assert cache.block_tables[0][0] == cache.block_tables[1][0]  # full prefix still shared
    assert cache.block_tables[0][1] != cache.block_tables[1][1]
    cache.free(0)
    cache.free(1)
    assert cache.stats().free_blocks == 8


def test_needs_new_block():
    cache = PagedKVCache(num_blocks=4, block_size=4)
    cache.add_sequence(0, 4)
    assert cache.needs_new_block(0)
    cache.append_token(0)
    assert not cache.needs_new_block(0)


def test_paging_wastes_far_less_than_contiguous_reservation():
    lens = [120, 300, 45, 800, 210]
    assert contiguous_utilization(lens, max_seq_len=4096) < 0.08
    assert paged_utilization(lens, block_size=16) > 0.97


def test_stats_utilization_counts_shared_blocks_once():
    cache = PagedKVCache(num_blocks=8, block_size=4)
    cache.add_sequence(0, 8)
    cache.fork(0, 1)
    stats = cache.stats()
    assert stats.tokens == 16 and stats.used_blocks == 2 and stats.slot_utilization == 1.0
