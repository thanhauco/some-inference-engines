import pytest

from inference_engines.concepts import parallelism, prefill_decode, routing
from inference_engines.concepts.batching import CostModel


# -- prefill / decode interference ----------------------------------------------------------
def test_monolithic_prefill_stalls_every_decoding_sequence():
    mono, chunked, disagg = prefill_decode.compare()
    baseline = CostModel().step_ms(32)
    assert mono.itl_p50_ms == pytest.approx(baseline)
    assert mono.itl_max_ms > 30 * baseline  # one 32k prompt freezes all streams
    assert chunked.itl_max_ms < mono.itl_max_ms / 10
    assert disagg.itl_max_ms == pytest.approx(baseline)  # decode GPU never sees a prompt


def test_chunking_trades_a_little_ttft_for_smooth_streams():
    mono, chunked, _ = prefill_decode.compare()
    assert mono.mean_prompt_ttft_ms < chunked.mean_prompt_ttft_ms < 2 * mono.mean_prompt_ttft_ms


def test_smaller_chunks_bound_the_spike_tighter():
    big = prefill_decode.simulate("chunked", chunk_tokens=8192)
    small = prefill_decode.simulate("chunked", chunk_tokens=512)
    assert small.itl_max_ms < big.itl_max_ms
    assert small.mean_prompt_ttft_ms > big.mean_prompt_ttft_ms


def test_no_prompts_means_no_interference():
    result = prefill_decode.simulate("monolithic", prompts=[])
    assert result.itl_max_ms == result.itl_p50_ms and result.mean_prompt_ttft_ms == 0
    with pytest.raises(ValueError):
        prefill_decode.simulate("magic")


# -- routing --------------------------------------------------------------------------------
def test_round_robin_destroys_prefix_reuse_across_replicas():
    results = {r.policy: r for r in routing.compare_policies()}
    assert results["round_robin"].hit_rate < 0.25
    assert results["prefix_affinity"].hit_rate > 3 * results["round_robin"].hit_rate
    assert results["consistent_hash"].hit_rate > 3 * results["round_robin"].hit_rate


def test_prefix_affinity_stays_balanced():
    results = {r.policy: r for r in routing.compare_policies()}
    assert results["prefix_affinity"].load_imbalance < 1.1
    assert results["round_robin"].load_imbalance == 1.0


def test_single_replica_makes_policies_equivalent():
    rates = {r.hit_rate for r in routing.compare_policies(replicas=1, capacity_tokens=10**6)}
    assert len(rates) == 1
    with pytest.raises(ValueError):
        routing.simulate_routing([[1, 2]], policy="random")


# -- tensor parallelism ---------------------------------------------------------------------
def test_tp_buys_latency_replicas_buy_throughput():
    plans = {p.tp: p for p in parallelism.plan_layouts(gpus=4)}
    assert plans[4].tpot_ms < plans[2].tpot_ms < plans[1].tpot_ms
    assert plans[1].throughput_tok_s > plans[2].throughput_tok_s > plans[4].throughput_tok_s


def test_slow_interconnect_makes_tp_a_loss():
    assert parallelism.tp_speedup(4, link=parallelism.NVLINK) > 2
    assert parallelism.tp_speedup(4, link=parallelism.ETHERNET) < 1


def test_a_model_that_does_not_fit_forces_tp():
    plans = {p.tp: p for p in parallelism.plan_layouts("llama-3.1-70b", gpus=4)}
    assert not plans[1].fits and plans[2].fits and plans[4].fits
    assert plans[1].row()["TPOT ms"] == "-"
