#!/usr/bin/env bash
# Same prompt, every engine that is running: compare answers and TTFT/TPOT side by side.
set -uo pipefail
PROMPT="${1:-Explain the difference between TTFT and TPOT in two sentences.}"
for engine in ollama vllm sglang tensorrt-llm; do
  echo "=== $engine"
  ie chat "$engine" "$PROMPT" --max-tokens 120 || echo "(not running)"
  echo
done
