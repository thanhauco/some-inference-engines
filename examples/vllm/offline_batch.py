"""Offline batch inference with vLLM's Python API (no server, maximum throughput).

The LLM engine continuously batches all prompts itself, so passing 1,000 prompts at once is
far faster than looping over them. Requires a GPU and `pip install vllm`.

    python examples/vllm/offline_batch.py
"""

import time

from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

topics = ["PagedAttention", "continuous batching", "speculative decoding", "FP8", "KV cache"]
conversations = [
    [{"role": "user", "content": f"Explain {topic} to a backend engineer in 3 sentences. ({i})"}]
    for i in range(200)
    for topic in topics
]

llm = LLM(model=MODEL, gpu_memory_utilization=0.90, max_model_len=8192, enable_prefix_caching=True)
params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

start = time.perf_counter()
outputs = llm.chat(conversations, params)
elapsed = time.perf_counter() - start

generated = sum(len(o.outputs[0].token_ids) for o in outputs)
print(outputs[0].outputs[0].text)
print(
    f"\n{len(outputs)} requests, {generated} tokens in {elapsed:.1f} s "
    f"-> {generated / elapsed:.0f} output tok/s"
)
