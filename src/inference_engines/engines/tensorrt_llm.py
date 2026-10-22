"""TensorRT-LLM: compile the model into an engine tuned for one NVIDIA GPU, then serve it.

Two workflows exist:

1. **LLM API / trtllm-serve** (recommended): point it at a Hugging Face model and it
   builds or loads an optimized runtime for you; quantization is a config option.
2. **Explicit build**: ``convert_checkpoint.py`` (per model family) turns HF weights into a
   TensorRT-LLM checkpoint, then ``trtllm-build`` compiles a GPU-specific engine (the
   "plan" in the diagram), with fused kernels and fixed maximum shapes.

Engines are specific to the GPU architecture *and* to the max batch/sequence sizes they
were built with - that is the price of the speed.
"""

from __future__ import annotations

DOCKER_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release"

QUANT_ALGOS = ("FP8", "W8A16", "W4A16", "W4A16_AWQ", "W8A8_SQ_PER_CHANNEL", "NVFP4")


def convert_checkpoint_command(
    model_dir: str,
    output_dir: str,
    *,
    dtype: str = "bfloat16",
    tp_size: int = 1,
    family: str = "llama",
) -> list[str]:
    script = f"examples/models/core/{family}/convert_checkpoint.py"
    return [
        "python", script,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--dtype", dtype,
        "--tp_size", str(tp_size),
    ]  # fmt: skip


def build_command(
    checkpoint_dir: str,
    output_dir: str,
    *,
    max_batch_size: int = 64,
    max_input_len: int = 4096,
    max_seq_len: int = 8192,
    max_num_tokens: int | None = None,
    gemm_plugin: str = "auto",
) -> list[str]:
    if max_input_len > max_seq_len:
        raise ValueError("max_input_len cannot exceed max_seq_len")
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", output_dir,
        "--gemm_plugin", gemm_plugin,
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_seq_len", str(max_seq_len),
    ]  # fmt: skip
    if max_num_tokens:
        cmd += ["--max_num_tokens", str(max_num_tokens)]
    return cmd


def launch_command(
    model: str,
    *,
    port: int = 8001,
    tp_size: int = 1,
    max_batch_size: int | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = None,
) -> list[str]:
    """``trtllm-serve``: OpenAI-compatible server over a HF model or a prebuilt engine dir."""
    cmd = [
        "trtllm-serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp_size",
        str(tp_size),
    ]
    if max_batch_size:
        cmd += ["--max_batch_size", str(max_batch_size)]
    if kv_cache_free_gpu_memory_fraction is not None:
        cmd += ["--kv_cache_free_gpu_memory_fraction", str(kv_cache_free_gpu_memory_fraction)]
    return cmd
