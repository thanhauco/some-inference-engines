"""Will it fit? Weights + KV cache arithmetic for picking a model, quantization and engine.

Memory is the first constraint you hit with every engine:

* With **Ollama** it decides which quantization (``q4_K_M``, ``q8_0``...) runs on a laptop.
* With **vLLM / SGLang** the VRAM left after the weights *is* the KV cache, and the KV
  cache decides how many concurrent sequences you can batch.
* With **TensorRT-LLM** it decides whether FP8/INT4 lets you drop from 2 GPUs to 1.

KV cache per token = 2 (K and V) x layers x kv_heads x head_dim x bytes per element.
Grouped-query attention (``kv_heads`` < attention heads) is why modern models fit so many
more tokens than older ones.
"""

from __future__ import annotations

from dataclasses import dataclass

GiB = 1024**3

BITS_PER_WEIGHT = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "fp8": 8.0,
    "int8": 8.0,
    "int4": 4.0,
    # llama.cpp / Ollama GGUF formats, effective bits including block scales.
    "q8_0": 8.5,
    "q6_K": 6.56,
    "q5_K_M": 5.69,
    "q4_K_M": 4.85,
    "q4_0": 4.5,
    "q3_K_M": 3.91,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int


MODELS = {
    "llama-3.1-8b": ModelSpec("Llama 3.1 8B", 8.03, 32, 8, 128, 131_072),
    "llama-3.1-70b": ModelSpec("Llama 3.1 70B", 70.6, 80, 8, 128, 131_072),
    "mistral-7b": ModelSpec("Mistral 7B v0.3", 7.25, 32, 8, 128, 32_768),
    "qwen2.5-7b": ModelSpec("Qwen2.5 7B", 7.62, 28, 4, 128, 131_072),
    "qwen2.5-32b": ModelSpec("Qwen2.5 32B", 32.8, 64, 8, 128, 131_072),
}


def weight_bytes(model: ModelSpec, quant: str = "bf16") -> float:
    return model.params_b * 1e9 * BITS_PER_WEIGHT[quant] / 8


def kv_bytes_per_token(model: ModelSpec, kv_dtype_bytes: float = 2) -> float:
    return 2 * model.layers * model.kv_heads * model.head_dim * kv_dtype_bytes


@dataclass(frozen=True)
class FitReport:
    model: str
    quant: str
    weights_gib: float
    kv_gib_per_seq: float
    max_concurrent_seqs: int
    fits: bool


def plan(
    model: ModelSpec | str,
    *,
    vram_gib: float,
    quant: str = "bf16",
    context_tokens: int = 8192,
    kv_dtype_bytes: float = 2,
    utilization: float = 0.9,
    overhead_gib: float = 1.5,
    num_gpus: int = 1,
) -> FitReport:
    """How many ``context_tokens``-long sequences fit next to the weights?

    ``utilization`` mirrors vLLM's ``--gpu-memory-utilization``; ``overhead_gib`` covers
    activations, CUDA graphs and the runtime itself (per GPU).
    """
    spec = MODELS[model] if isinstance(model, str) else model
    usable = vram_gib * utilization * num_gpus - overhead_gib * num_gpus
    weights = weight_bytes(spec, quant) / GiB
    per_seq = kv_bytes_per_token(spec, kv_dtype_bytes) * context_tokens / GiB
    free = usable - weights
    seqs = int(free // per_seq) if free > 0 else 0
    return FitReport(spec.name, quant, round(weights, 2), round(per_seq, 3), seqs, seqs >= 1)


def best_quant_for(
    model: ModelSpec | str,
    vram_gib: float,
    context_tokens: int = 8192,
    min_seqs: int = 1,
    options: tuple[str, ...] = ("bf16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q3_K_M"),
) -> str | None:
    """Highest-precision format that leaves room for ``min_seqs`` sequences."""
    for quant in options:
        report = plan(model, vram_gib=vram_gib, quant=quant, context_tokens=context_tokens)
        if report.max_concurrent_seqs >= min_seqs:
            return quant
    return None
