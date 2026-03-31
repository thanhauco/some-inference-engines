# vLLM - scalable production serving

> **Choose it when** you are building a production API and care about concurrency,
> throughput and GPU utilization without going too deep into hardware-specific tuning.

## The two ideas that made vLLM

### 1. PagedAttention (memory)

Before vLLM, servers reserved one contiguous KV region per request, sized for the
maximum sequence length. A 300-token chat in a 4,096-token slot wastes 93% of it, and
that waste limits the batch.

PagedAttention stores KV in fixed-size **blocks** (16 tokens by default) and keeps a
**block table** per sequence, exactly like an OS page table:

```
seq A: [blk 7][blk 2][blk 9]      physical KV memory
seq B: [blk 7][blk 4]             ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
         ▲ shared prefix block    │  │  │A │  │B │  │  │AB│  │A │
                                  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

* Only the last block of each sequence is partially empty → < 4% waste.
* Blocks are reference counted → parallel samples, beams and shared prefixes share
  memory; a write to a shared block triggers copy-on-write.

Code: [`concepts/kv_cache.py`](../src/inference_engines/concepts/kv_cache.py).
`ie simulate paged` shows 17% vs 99% memory utilization on the same sequences.

### 2. Continuous batching (scheduling)

Static batching runs a batch until its *longest* request finishes; short requests leave
holes and new requests wait. vLLM schedules at **iteration level**: after every decode
step, finished sequences exit and waiting ones join, as long as the KV cache has free
blocks. When memory runs out mid-generation, the newest sequence is **preempted** (its
blocks freed and recomputed later).

Code: [`concepts/batching.py`](../src/inference_engines/concepts/batching.py).
`ie simulate batching` on the same workload and memory budget:

| scheduler | throughput | mean TTFT | slot utilization | max batch |
|---|---|---|---|---|
| static | ~1,000 tok/s | ~12.5 s | 55% | 48 |
| continuous + paged | ~2,000 tok/s | ~45 ms | 100% | 86 |

## Everything else you get

* OpenAI-compatible server (`vllm serve`), offline `LLM` API for batch jobs.
* Automatic prefix caching, chunked prefill, speculative decoding (EAGLE, n-gram, MTP),
  structured outputs, multi-LoRA, multimodal models.
* Tensor, pipeline, data and expert parallelism.
* Broad hardware: NVIDIA, AMD ROCm, and plugins for other accelerators (TPU, Gaudi, ...).
* Prometheus `/metrics` - see [`examples/vllm/watch_scheduler.py`](../examples/vllm/watch_scheduler.py).

## The knobs that matter

| Flag | Effect | Turn it when |
|---|---|---|
| `--gpu-memory-utilization` | VRAM share for vLLM; leftover after weights = KV cache | OOM at startup (down) / idle VRAM (up) |
| `--max-model-len` | Longest context | KV cache too small for your concurrency |
| `--max-num-seqs` | Batch size cap | Trade p99 latency for throughput |
| `--enable-prefix-caching` | Reuse KV across shared prefixes | Long system prompts, RAG templates, multi-turn |
| `--tensor-parallel-size` | Shard across GPUs | Model doesn't fit, or need lower latency |
| `--quantization fp8` | Smaller weights, faster decode | Memory-bound, Hopper+ GPUs |

The signal to watch: `num_requests_waiting > 0` while `kv_cache_usage` ≈ 100% means
you are memory-bound - shorten context, quantize, or add GPUs.

## Strengths and limits

| Pros | Cons |
|---|---|
| High throughput via continuous batching | More setup than Ollama |
| Memory efficient (PagedAttention) | Focused on serving, not every last % of hardware tuning |
| Long-context support, wide model coverage | Best on NVIDIA GPUs |
| Large, active open-source community | |

See [vllm-vs-sglang.md](vllm-vs-sglang.md) for a deep comparison with SGLang.
