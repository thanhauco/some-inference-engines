#!/usr/bin/env bash
# The explicit two-step build: HF weights -> TensorRT-LLM checkpoint -> GPU-specific engine.
# Run inside the TensorRT-LLM container (see deploy/docker-compose.yml) from a clone of
# https://github.com/NVIDIA/TensorRT-LLM so the per-model convert scripts are available.
set -euo pipefail
HF_MODEL_DIR="${HF_MODEL_DIR:-/models/Llama-3.1-8B-Instruct}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/llama-3.1-8b-bf16-tp1}"
ENGINE_DIR="${ENGINE_DIR:-./engines/llama-3.1-8b-bf16-tp1}"

# 1. Convert: re-layout and (optionally) quantize weights, shard for tensor parallelism.
python examples/models/core/llama/convert_checkpoint.py \
  --model_dir "$HF_MODEL_DIR" --output_dir "$CKPT_DIR" --dtype bfloat16 --tp_size 1

# 2. Build: TensorRT fuses layers/kernels and picks the fastest tactics for *this* GPU.
#    The max_* values are baked in - requests beyond them are rejected.
trtllm-build \
  --checkpoint_dir "$CKPT_DIR" --output_dir "$ENGINE_DIR" \
  --gemm_plugin auto \
  --max_batch_size 128 --max_input_len 8192 --max_seq_len 16384 --max_num_tokens 8192

# 3. Serve the engine (its directory holds the "plan" files).
echo "trtllm-serve $ENGINE_DIR --tokenizer $HF_MODEL_DIR --port 8001"
