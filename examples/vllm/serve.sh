#!/usr/bin/env bash
# OpenAI-compatible vLLM server. Every flag here is a throughput/memory decision.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# --gpu-memory-utilization  VRAM vLLM may take; what remains after weights = paged KV cache
# --max-model-len           cap context so more sequences fit in the KV cache
# --max-num-seqs            upper bound on the continuous batch
# --enable-prefix-caching   reuse KV blocks of shared prompt prefixes (default on in recent versions)
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --tensor-parallel-size "${TP:-1}"
