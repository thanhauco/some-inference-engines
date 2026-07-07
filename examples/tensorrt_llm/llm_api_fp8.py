"""TensorRT-LLM's high-level LLM API with FP8 quantization.

FP8 halves weight memory and runs matmuls on FP8 tensor cores (Hopper/Ada/Blackwell).
Requires an NVIDIA GPU with FP8 support and `pip install tensorrt-llm` (or the NGC image).

    python examples/tensorrt_llm/llm_api_fp8.py
"""

import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import QuantAlgo, QuantConfig


def main() -> None:
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8),
    )
    prompts = [f"Give one tip for lowering LLM latency (#{i})." for i in range(64)]
    params = SamplingParams(max_tokens=128, temperature=0.0)

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(outputs[0].outputs[0].text)
    print(f"\n{tokens} tokens in {elapsed:.2f} s -> {tokens / elapsed:.0f} tok/s")


if __name__ == "__main__":  # TensorRT-LLM spawns workers; keep the guard
    main()
