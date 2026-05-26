#!/usr/bin/env bash
# SGLang runtime. RadixAttention (automatic prefix reuse) is on by default.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

ARGS=(--model-path "$MODEL" --host 0.0.0.0 --port 30000 --tp "${TP:-1}" --mem-fraction-static 0.85)

# Optional: EAGLE speculative decoding (draft head proposes, target verifies).
# DRAFT=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B ./examples/sglang/serve.sh
if [[ -n "${DRAFT:-}" ]]; then
  ARGS+=(--speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT"
         --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi

exec python -m sglang.launch_server "${ARGS[@]}"
