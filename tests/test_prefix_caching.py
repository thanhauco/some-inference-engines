import pytest

from inference_engines.concepts.prefix_caching import (
    BlockHashCache,
    RadixPrefixCache,
    grouped_workload,
    head_to_head,
    simulate_queue,
)


def test_block_hash_cache_only_shares_full_blocks():
    cache = BlockHashCache(capacity_tokens=1000, block_size=16)
    shared = list(range(40))  # 2 full blocks + 8 tokens
    assert cache.serve(shared + [100]) == 41
    assert cache.match_prefix(shared + [200]) == 32  # the partial third block is recomputed
    assert cache.serve(shared + [200]) == 9


def test_radix_cache_shares_to_the_token():
    cache = RadixPrefixCache(capacity_tokens=1000)
    shared = list(range(40))
    cache.serve(shared + [100])
    assert cache.match_prefix(shared + [200]) == 40


def test_block_hashes_are_chained_to_the_parent():
    cache = BlockHashCache(capacity_tokens=1000, block_size=4)
    cache.serve([1, 2, 3, 4, 5, 6, 7, 8])
    # Same second block, different first block -> different prefix -> no hit at all.
    assert cache.match_prefix([9, 9, 9, 9, 5, 6, 7, 8]) == 0


def test_lru_eviction_bounds_capacity():
    cache = BlockHashCache(capacity_tokens=64, block_size=16)
    for start in range(0, 1000, 100):
        cache.serve(list(range(start, start + 64)))
    assert len(cache._blocks) == 4
    assert cache.match_prefix(list(range(900, 964))) == 64
    assert cache.match_prefix(list(range(0, 64))) == 0


def test_cache_aware_scheduling_beats_fcfs_under_memory_pressure():
    results = {(r.cache, r.policy): r for r in head_to_head(capacity_tokens=3000)}
    for cache in ("block-hash (vLLM)", "radix (SGLang)"):
        assert results[(cache, "lpm")].hit_rate > 3 * results[(cache, "fcfs")].hit_rate


def test_with_ample_memory_policy_stops_mattering_and_caches_tie():
    results = head_to_head(capacity_tokens=1_000_000)
    rates = [r.hit_rate for r in results]
    assert max(rates) - min(rates) < 0.02


def test_unknown_policy():
    with pytest.raises(ValueError):
        simulate_queue(grouped_workload(2, 2), RadixPrefixCache(100), policy="random")
