"""GPU/CPU layer offload and cold starts: the two things that shape Ollama's performance.

**Partial offload.** Ollama (llama.cpp) puts as many transformer layers on the GPU as fit
in VRAM and runs the rest on the CPU. Every token passes through *all* layers, so the
slow CPU layers dominate: token time = gpu_layers x t_gpu + cpu_layers x t_cpu. Speed
does not degrade gracefully - with a quarter of the layers on the CPU it drops by more
than half. That is why a quantization that fits *entirely* in VRAM (see :mod:`.memory`)
usually beats a higher-precision one that spills.

**Cold starts.** A model is loaded on first use and unloaded after ``OLLAMA_KEEP_ALIVE``
(5 minutes by default). The first request after an idle period pays the load time:
file size / disk throughput. Great for a laptop that needs its memory back; a latency
trap for anything that looks like a service.

Default layer timings are illustrative (a mid-range laptop GPU vs its CPU); pass your
own measurements to model your machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .memory import MODELS, GiB, ModelSpec, kv_bytes_per_token, weight_bytes


@dataclass
class OffloadPlan:
    model: str
    quant: str
    vram_gib: float
    gpu_layers: int
    total_layers: int
    tokens_per_s: float

    @property
    def fully_offloaded(self) -> bool:
        return self.gpu_layers == self.total_layers

    def row(self) -> dict[str, float | str]:
        return {
            "VRAM GiB": self.vram_gib,
            "quant": self.quant,
            "GPU layers": f"{self.gpu_layers}/{self.total_layers}",
            "tok/s": round(self.tokens_per_s, 1),
        }


def gpu_layers_that_fit(
    model: ModelSpec | str,
    quant: str,
    vram_gib: float,
    *,
    context_tokens: int = 4096,
    overhead_gib: float = 1.0,
) -> int:
    """How many layers (weights + their share of the KV cache) fit in ``vram_gib``."""
    spec = MODELS[model] if isinstance(model, str) else model
    per_layer = (
        weight_bytes(spec, quant) + kv_bytes_per_token(spec) * context_tokens
    ) / spec.layers
    usable = (vram_gib - overhead_gib) * GiB
    return max(0, min(spec.layers, int(usable // per_layer)))


def plan_offload(
    model: ModelSpec | str = "llama-3.1-8b",
    quant: str = "q4_K_M",
    vram_gib: float = 8.0,
    *,
    gpu_ms_per_layer: float = 0.5,
    cpu_ms_per_layer: float = 4.0,
    context_tokens: int = 4096,
) -> OffloadPlan:
    spec = MODELS[model] if isinstance(model, str) else model
    # Layer cost scales with bytes per layer: higher-precision weights are slower to read.
    scale = weight_bytes(spec, quant) / weight_bytes(MODELS["llama-3.1-8b"], "q4_K_M")
    scale *= MODELS["llama-3.1-8b"].layers / spec.layers
    on_gpu = gpu_layers_that_fit(spec, quant, vram_gib, context_tokens=context_tokens)
    on_cpu = spec.layers - on_gpu
    token_ms = (on_gpu * gpu_ms_per_layer + on_cpu * cpu_ms_per_layer) * scale
    return OffloadPlan(spec.name, quant, vram_gib, on_gpu, spec.layers, 1000 / token_ms)


def offload_curve(
    model: ModelSpec | str = "llama-3.1-8b",
    quant: str = "q4_K_M",
    vram_options: tuple[float, ...] = (0, 2, 4, 6, 8, 12),
    **kwargs,
) -> list[OffloadPlan]:
    return [plan_offload(model, quant, vram, **kwargs) for vram in vram_options]


def recommend_quant(
    model: ModelSpec | str,
    vram_gib: float,
    options: tuple[str, ...] = ("q8_0", "q6_K", "q5_K_M", "q4_K_M", "q3_K_M"),
    **kwargs,
) -> OffloadPlan:
    """Highest-precision quantization that runs entirely on the GPU; else the fastest split.

    ``options`` must be ordered from highest to lowest precision.
    """
    plans = [plan_offload(model, quant, vram_gib, **kwargs) for quant in options]
    for plan in plans:
        if plan.fully_offloaded:
            return plan
    return max(plans, key=lambda plan: plan.tokens_per_s)


def cold_start_s(
    model: ModelSpec | str, quant: str = "q4_K_M", disk_gb_per_s: float = 2.0
) -> float:
    """Seconds to load the model file from disk (NVMe ~2-3 GB/s, SATA SSD ~0.5)."""
    spec = MODELS[model] if isinstance(model, str) else model
    return weight_bytes(spec, quant) / 1e9 / disk_gb_per_s


def expected_cold_starts_per_hour(requests_per_hour: float, keep_alive_min: float = 5.0) -> float:
    """Cold starts/hour for Poisson traffic: requests arriving after a gap > keep-alive."""
    if requests_per_hour <= 0:
        return 0.0
    rate_per_min = requests_per_hour / 60
    return requests_per_hour * math.exp(-rate_per_min * keep_alive_min)
