#!/usr/bin/env bash
# trtllm-serve: OpenAI-compatible server on top of TensorRT-LLM.
# MODEL may be a Hugging Face id (built/optimized on load) or a prebuilt engine directory.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

exec trtllm-serve "$MODEL" \
  --host 0.0.0.0 --port 8001 \
  --tp_size "${TP:-1}" \
  --max_batch_size 128 \
  --kv_cache_free_gpu_memory_fraction 0.90
