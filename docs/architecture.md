# The four pipelines, step by step

Every engine runs the same high-level loop: **request in → schedule → prefill → decode
tokens → stream out**. They differ in *which step they optimize* and for whom. This page
walks the diagram column by column and points to the code in this repo that simulates
each step.

```
             vLLM                 SGLang                 TensorRT-LLM            Ollama
         ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐   ┌──────────────┐
  1      │ many users   │    │ many users       │    │ load + compile    │   │ one user     │
         ├──────────────┤    ├──────────────────┤    ├───────────────────┤   ├──────────────┤
  2      │ continuous   │    │ program-based    │    │ optimized engine  │   │ load model   │
         │ batching     │    │ requests         │    │ (fused, quantized)│   │ from registry│
         ├──────────────┤    ├──────────────────┤    ├───────────────────┤   ├──────────────┤
  3      │ PagedAttn    │    │ speculative /    │    │ batched inference │   │ run on CPU / │
         │ (KV pages)   │    │ parallel decoding│    │ (in-flight)       │   │ GPU / Metal  │
         ├──────────────┤    ├──────────────────┤    ├───────────────────┤   ├──────────────┤
  4      │ decode       │    │ optimized runtime│    │ decode w/ TRT     │   │ generate     │
         │ tokens       │    │ (scheduling)     │    │ kernels           │   │ tokens       │
         ├──────────────┤    ├──────────────────┤    ├───────────────────┤   ├──────────────┤
  5      │ stream to    │    │ stream to agents │    │ ultra-low-latency │   │ return to    │
         │ many users   │    │ / tools          │    │ responses         │   │ CLI/API/app  │
         └──────────────┘    └──────────────────┘    └───────────────────┘   └──────────────┘
```

## Background: why serving LLMs is a memory problem

Generating a token has two phases:

* **Prefill** processes the whole prompt in parallel. It is *compute-bound* and decides
  **TTFT** (time to first token).
* **Decode** produces one token per sequence per step. Each step streams *all* model
  weights from GPU memory (HBM) to compute a handful of tokens, so it is
  *memory-bandwidth-bound* and decides **TPOT** (time per output token).

Two consequences drive every design below:

1. A decode step over 64 sequences costs barely more than over 1, so **batching is
   nearly free throughput** - if you can keep the batch full.
2. Every in-flight sequence holds its **KV cache** (keys/values of all previous tokens)
   in GPU memory - 128 KiB per token for Llama-3.1-8B in BF16. KV memory, not compute,
   usually caps the batch size. (`ie simulate memory`)

## vLLM - high-throughput serving for production LLMs

| Step | What happens | In this repo |
|---|---|---|
| 1. Many users | Concurrent requests arrive at an OpenAI-compatible server. | `client.py` |
| 2. Continuous batching | The scheduler re-forms the batch **every step**: finished sequences leave, waiting ones join immediately. No request waits for the longest one in its batch. | `concepts/batching.py` |
| 3. PagedAttention | KV cache is split into fixed-size **blocks** mapped through per-sequence block tables, like virtual memory pages. Waste drops from ~60-80% (contiguous max-length reservations) to under 4%, so many more sequences fit. Shared prefixes share blocks (prefix caching, copy-on-write forks). | `concepts/kv_cache.py`, `concepts/prefix_caching.py` |
| 4. Decode | Optimized attention kernels (FlashAttention / FlashInfer), CUDA graphs, chunked prefill mixing prompt and decode work in one step. | - |
| 5. Stream | Server-sent events per token to every client. | `client.py` (`stream_chat`) |

Run `ie simulate batching` and `ie simulate paged` to watch steps 2 and 3 pay off.

## SGLang - programmable, high-performance serving

| Step | What happens | In this repo |
|---|---|---|
| 1. Many users | Same OpenAI-compatible entry point, plus a native API for programs. | `engines/sglang.py` |
| 2. Program-based requests | A request can be a *program*: `gen`, `select`, `fork`, regex/JSON constraints, tool calls, branches. The frontend language expresses them; the runtime executes them efficiently. | `examples/sglang/agent_program.py` |
| 3. Speculative / parallel decoding | A draft (EAGLE head) proposes several tokens, the target verifies them in one pass. `fork` runs branches in parallel over a shared prefix. | `concepts/speculative.py` |
| 4. Optimized runtime | **RadixAttention** keeps KV of past requests in a radix tree so any shared prefix is reused; **cache-aware scheduling** serves cached prefixes first; grammar masks with **jump-forward** decoding for structured output; CPU scheduling overlapped with GPU work. | `concepts/radix_cache.py`, `concepts/prefix_caching.py`, `concepts/constrained.py` |
| 5. Stream | Streams to agents and tools; structured outputs parse by construction. | `examples/sglang/structured_output.py` |

## TensorRT-LLM - maximum performance with NVIDIA optimizations

| Step | What happens | In this repo |
|---|---|---|
| 1. Load and optimize | HF/ONNX weights are converted and **compiled** into a TensorRT engine ("plan") specific to the GPU architecture and max shapes - or built automatically by the LLM API / `trtllm-serve`. | `engines/tensorrt_llm.py`, `examples/tensorrt_llm/build_engine.sh` |
| 2. Optimized execution | **Kernel fusion** (bias+activation+residual in one kernel), **layer fusion** (QKV as one GEMM), **quantization** (FP8, INT8, INT4 AWQ, NVFP4), memory optimizations (paged KV, KV quantization). | `concepts/fusion.py`, `concepts/quantization.py` |
| 3. Batched inference | In-flight batching (TensorRT-LLM's continuous batching), multi-stream execution, tuned GEMM tactics. | `concepts/batching.py` |
| 4. Generate | Autoregressive decoding with TensorRT kernels; speculative decoding variants too. | - |
| 5. Stream | Lowest latency per token on NVIDIA hardware. | `examples/tensorrt_llm/serve.sh` |

## Ollama - simple local LLM running on your machine

| Step | What happens | In this repo |
|---|---|---|
| 1. Local request | One user (or a few) from the CLI, an app or the API on `localhost:11434`. | `engines/ollama.py` |
| 2. Load from registry | `ollama pull llama3.1:8b` downloads a quantized GGUF once; the model is loaded on first use and unloaded after idling. | `examples/ollama/quickstart.sh` |
| 3. Run locally | llama.cpp underneath: CPU, NVIDIA/AMD GPU or Apple Silicon (Metal), splitting layers between GPU and CPU if VRAM is short. | `concepts/memory.py` (which quantization fits) |
| 4. Generate | Straightforward token generation, modest parallelism. | - |
| 5. Return | Text to the CLI, API or app. | `examples/ollama/native_api.py` |

## The common denominator: the OpenAI API

All four expose `POST /v1/chat/completions` with SSE streaming. That one fact is what
makes the migration path in [migration-path.md](migration-path.md) cheap: the model and
the application can stay the same while the engine changes underneath.
