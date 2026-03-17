import pytest

from inference_engines.concepts.batching import (
    CostModel,
    Request,
    compare,
    make_workload,
    simulate_continuous,
    simulate_static,
)


def test_every_request_completes_with_all_its_tokens():
    workload = make_workload(60, seed=3)
    for result in compare(workload, kv_capacity_tokens=60_000):
        assert sorted(r.id for r in result.results) == list(range(60))
        produced = {r.id: r.output_tokens for r in result.results}
        assert produced == {r.id: r.output_tokens for r in workload}
        assert all(r.first_token_ms <= r.finish_ms for r in result.results)


def test_continuous_batching_beats_static_on_throughput_and_ttft():
    static, continuous = compare(make_workload(150, seed=1))
    assert continuous.throughput_tok_s > 1.5 * static.throughput_tok_s
    assert continuous.mean_ttft_ms < static.mean_ttft_ms / 10
    assert continuous.slot_utilization == 1.0
    assert static.slot_utilization < 0.8  # padding: finished slots wait for the longest request


def test_paging_fits_more_sequences_in_the_same_memory():
    static, continuous = compare(
        make_workload(150, rate_per_s=50, seed=2), kv_capacity_tokens=100_000
    )
    assert static.max_batch_seen <= 100_000 // 4096
    assert continuous.max_batch_seen > 2 * static.max_batch_seen


def test_preemption_under_memory_pressure_still_finishes():
    workload = [Request(i, 0.0, 100, 200) for i in range(10)]
    result = simulate_continuous(workload, kv_capacity_tokens=1_200, block_size=16)
    assert result.preemptions > 0
    assert len(result.results) == 10


def test_impossible_requests_are_rejected():
    with pytest.raises(ValueError):
        simulate_continuous([Request(0, 0, 900, 200)], kv_capacity_tokens=1_000)
    with pytest.raises(ValueError):
        simulate_static([Request(0, 0, 10, 10)], kv_capacity_tokens=100, max_seq_len=4096)


def test_cost_model_is_memory_bound():
    cost = CostModel()
    assert cost.step_ms(64) < 2 * cost.step_ms(1)  # 64x the tokens for <2x the time
