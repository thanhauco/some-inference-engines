import pytest

from inference_engines.concepts.memory import MODELS, best_quant_for, kv_bytes_per_token, plan


def test_llama3_8b_kv_cache_is_128_kib_per_token():
    assert kv_bytes_per_token(MODELS["llama-3.1-8b"]) == 128 * 1024


def test_8b_on_an_80gb_gpu_serves_dozens_of_8k_sequences():
    report = plan("llama-3.1-8b", vram_gib=80, context_tokens=8192)
    assert report.weights_gib == pytest.approx(14.96, abs=0.05)
    assert 40 < report.max_concurrent_seqs < 70


def test_70b_needs_quantization_or_more_gpus():
    assert not plan("llama-3.1-70b", vram_gib=80).fits
    assert plan("llama-3.1-70b", vram_gib=80, quant="fp8").fits
    assert plan("llama-3.1-70b", vram_gib=80, num_gpus=2).fits


def test_best_quant_for_laptops():
    assert best_quant_for("llama-3.1-8b", vram_gib=8) == "q4_K_M"
    assert best_quant_for("llama-3.1-8b", vram_gib=24) == "bf16"
    assert best_quant_for("llama-3.1-70b", vram_gib=24) is None
