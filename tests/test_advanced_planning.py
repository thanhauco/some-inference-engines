import math

import pytest

from inference_engines.concepts import capacity, offload


# -- Ollama offload -------------------------------------------------------------------------
def test_offload_is_a_cliff_not_a_slope():
    full = offload.plan_offload(vram_gib=12)
    cpu = offload.plan_offload(vram_gib=0)
    assert full.fully_offloaded and cpu.gpu_layers == 0
    assert full.tokens_per_s == pytest.approx(8 * cpu.tokens_per_s)
    half = offload.plan_offload(vram_gib=4)
    assert 0 < half.gpu_layers < 32
    assert half.tokens_per_s < full.tokens_per_s / 2  # most layers on GPU, most speed gone


def test_recommend_quant_prefers_precision_that_fits_entirely():
    assert offload.recommend_quant("llama-3.1-8b", 8).quant == "q6_K"
    assert offload.recommend_quant("llama-3.1-8b", 16).quant == "q8_0"
    spilled = offload.recommend_quant("llama-3.1-70b", 8)
    assert not spilled.fully_offloaded


def test_cold_starts():
    assert offload.cold_start_s("llama-3.1-8b", disk_gb_per_s=2.0) == pytest.approx(2.43, abs=0.01)
    rare, busy = (
        offload.expected_cold_starts_per_hour(6),
        offload.expected_cold_starts_per_hour(120),
    )
    assert rare > 3 and busy < 0.01  # sparse traffic pays the load almost every time
    assert offload.expected_cold_starts_per_hour(0) == 0


# -- capacity and cost ----------------------------------------------------------------------
def test_cost_per_million_tokens():
    assert capacity.cost_per_million_tokens(3.6, 1000) == pytest.approx(1.0)
    assert capacity.cost_per_million_tokens(3.6, 1000, utilization=0.25) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        capacity.cost_per_million_tokens(3.6, 0)


def test_fleet_sizing_and_breakeven():
    assert capacity.gpus_needed(50_000, 2000, target_utilization=0.7) == 36
    slow, _, fast = capacity.compare_options(capacity.EXAMPLE_MEASUREMENTS, peak_tok_per_s=50_000)
    assert fast.gpus < slow.gpus and fast.usd_per_million_tokens < slow.usd_per_million_tokens
    months = capacity.breakeven_months(60_000, slow, fast)
    assert months == pytest.approx(60_000 / (slow.monthly_usd - fast.monthly_usd))
    assert capacity.breakeven_months(60_000, fast, slow) == math.inf
