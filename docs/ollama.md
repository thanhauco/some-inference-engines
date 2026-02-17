# Ollama - the simplest local experience

> **Choose it when** you want the fastest path to local experimentation and
> production-scale concurrency is not yet your problem.

## What it is

A single binary that bundles a model registry, a background server (`ollama serve`,
port 11434) and [llama.cpp](https://github.com/ggml-org/llama.cpp) as the inference
backend. Models are distributed as **GGUF** files, typically 4-bit quantized, so an 8B
model is ~5 GB and runs on a laptop.

```bash
ollama pull llama3.1:8b          # once
ollama run llama3.1:8b "hello"   # loads into RAM/VRAM on first use
ollama ps                        # what is loaded, and the CPU/GPU split
```

## How it works

* **Registry + Modelfile.** `pull` fetches layers like a container image. A
  [Modelfile](../examples/ollama/Modelfile) layers a system prompt and parameters on top
  of a base model (`ollama create my-bot -f Modelfile`).
* **Lazy load / unload.** A model is loaded on its first request and evicted after an
  idle timeout (`OLLAMA_KEEP_ALIVE`, 5 minutes by default) - great for a laptop that
  runs other things, and the reason the first call after a pause is slow.
* **Hardware fallback.** If the model doesn't fit in VRAM, llama.cpp offloads some layers
  to the GPU and runs the rest on CPU. It still works, just slower.
* **Some parallelism.** `OLLAMA_NUM_PARALLEL` lets one loaded model serve several
  requests at once, but each slot reserves its own context window, and there is no
  PagedAttention-style memory sharing or throughput-oriented scheduling.

## Which quantization fits?

`ie simulate memory` and `concepts/memory.py`:

| Machine | Llama 3.1 8B | Llama 3.1 70B |
|---|---|---|
| 8 GB laptop GPU / 16 GB unified memory | `q4_K_M` | no |
| 24 GB GPU / 32 GB Mac | `q8_0` or BF16 | no |
| 64 GB+ Mac | BF16 | `q4_K_M` |

Rule of thumb: prefer the highest precision that leaves room for your context window;
`q4_K_M` is the usual sweet spot of quality vs. size.

## Strengths and limits

| Pros | Cons |
|---|---|
| One-command setup, works offline | Not designed for high-concurrency serving |
| CPU, NVIDIA, AMD, Apple Silicon | Limited to one machine's resources |
| Huge library of ready-to-run models | Lower throughput than vLLM/SGLang/TensorRT-LLM on the same GPU |
| Great for privacy-sensitive and offline work | Fewer knobs for latency/throughput tuning |

## In this repo

* [`engines/ollama.py`](../src/inference_engines/engines/ollama.py) - native API client
  (`pull`, `ensure_model`, `/api/ps`, decode tok/s from Ollama's own counters).
* [`examples/ollama/`](../examples/ollama) - quickstart script, Modelfile, native API demo.
* The OpenAI-compatible endpoint (`/v1`) is what lets you move to another engine later
  without changing application code.
