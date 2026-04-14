from inference_engines.concepts.radix_cache import RadixCache, agent_workload, prefill_savings


def test_match_and_split_edges():
    cache = RadixCache()
    assert cache.insert([1, 2, 3, 4]) == 4
    assert cache.match_prefix([1, 2, 3, 4, 5]) == 4
    assert cache.match_prefix([1, 2, 9]) == 2
    assert cache.insert([1, 2, 9, 9]) == 2  # splits the [1,2,3,4] edge at 2
    assert cache.total_tokens == 6
    assert cache.num_nodes() == 3  # [1,2] -> {[3,4], [9,9]}
    assert cache.match_prefix([1, 2, 3, 4]) == 4
    assert cache.match_prefix([7]) == 0


def test_inserting_a_prefix_of_an_existing_entry_adds_nothing():
    cache = RadixCache()
    cache.insert([1, 2, 3, 4])
    assert cache.insert([1, 2]) == 0
    assert cache.match_prefix([1, 2, 3]) == 3


def test_serve_counts_reuse():
    cache = RadixCache()
    system = list(range(100))
    assert cache.serve(system + [1000, 1001]) == 102
    assert cache.serve(system + [2000]) == 1  # only the new tail is prefilled
    assert cache.counters.hit_rate == 100 / 203


def test_lru_eviction_keeps_hot_prefix():
    cache = RadixCache(capacity_tokens=130)
    system = list(range(100))
    cache.serve(system + [1000] * 10)
    cache.serve(system + [2000] * 10)
    cache.serve(system + [1000] * 10)  # session 1 is hot again
    cache.serve(system + [3000] * 20)  # over capacity -> evict LRU leaf (session 2)
    assert cache.total_tokens <= 130
    assert cache.match_prefix(system + [2000] * 10) == 100
    assert cache.match_prefix(system + [3000] * 20) == 120
    assert cache.counters.evicted_tokens >= 10


def test_agent_workload_gets_order_of_magnitude_prefill_reduction():
    prompts = agent_workload(num_sessions=8, turns=5)
    assert all(p[:1500] == list(range(1500)) for p in prompts)
    stats = prefill_savings(prompts)
    assert stats["hit_rate"] > 0.9
    assert stats["prefill_reduction_x"] > 10
    assert prefill_savings(prompts, capacity_tokens=5000)["prefill_reduction_x"] > 3
