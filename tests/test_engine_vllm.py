import pytest

from inference_engines.engines import vllm


def test_launch_command_flags():
    cmd = vllm.launch_command(
        "m", tensor_parallel_size=2, max_model_len=8192, max_num_seqs=128, quantization="fp8"
    )
    assert cmd[:3] == ["vllm", "serve", "m"]
    joined = " ".join(cmd)
    for flag in [
        "--tensor-parallel-size 2",
        "--max-model-len 8192",
        "--max-num-seqs 128",
        "--enable-prefix-caching",
        "--quantization fp8",
    ]:
        assert flag in joined


def test_launch_command_validates_memory_fraction():
    with pytest.raises(ValueError):
        vllm.launch_command("m", gpu_memory_utilization=1.5)


METRICS = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 42.0
vllm:num_requests_waiting{model_name="m"} 7.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.97
vllm:generation_tokens_total{model_name="m"} 1.5e4
"""


def test_scheduler_snapshot_reads_old_and_new_gauge_names():
    snap = vllm.scheduler_snapshot(METRICS)
    assert snap == {"running": 42.0, "waiting": 7.0, "kv_cache_usage": 0.97}
    renamed = METRICS.replace("gpu_cache_usage_perc", "kv_cache_usage_perc")
    assert vllm.scheduler_snapshot(renamed)["kv_cache_usage"] == 0.97
    assert vllm.parse_prometheus(METRICS)["vllm:generation_tokens_total"] == 15000.0
