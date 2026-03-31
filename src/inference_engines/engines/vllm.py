"""vLLM: high-throughput serving built on continuous batching and PagedAttention.

The flags below are the ones that decide throughput and memory in practice:

``--gpu-memory-utilization``  fraction of VRAM vLLM may claim; what is left after the
                              weights becomes the paged KV cache.
``--max-model-len``           longest sequence; lower it to fit more concurrent sequences.
``--max-num-seqs``            cap on sequences in one continuous batch.
``--enable-prefix-caching``   reuse KV blocks of shared prompt prefixes across requests.
``--tensor-parallel-size``    shard the model across N GPUs.
``--quantization``            e.g. ``fp8``, ``awq``, ``gptq``.
"""

from __future__ import annotations

import re

DOCKER_IMAGE = "vllm/vllm-openai"


def launch_command(
    model: str,
    *,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool = True,
    quantization: str | None = None,
    served_model_name: str | None = None,
) -> list[str]:
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]  # fmt: skip
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if quantization:
        cmd += ["--quantization", quantization]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    return cmd


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum Prometheus samples by metric name (labels are folded together)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line.strip())
        if match:
            name, _, value = match.groups()
            values[name] = values.get(name, 0.0) + float(value)
    return values


# vLLM renamed a few gauges across releases; accept either spelling.
_GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}


def scheduler_snapshot(metrics_text: str) -> dict[str, float | None]:
    """Extract the three numbers worth watching from vLLM's ``/metrics`` endpoint.

    ``waiting`` > 0 while ``kv_cache_usage`` is near 1.0 means the KV cache, not compute,
    is limiting concurrency: lower ``--max-model-len``, quantize, or add GPUs.
    """
    metrics = parse_prometheus(metrics_text)
    snapshot: dict[str, float | None] = {}
    for key, names in _GAUGES.items():
        snapshot[key] = next((metrics[n] for n in names if n in metrics), None)
    return snapshot
