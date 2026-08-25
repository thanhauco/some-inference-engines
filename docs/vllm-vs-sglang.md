# vLLM vs SGLang, in depth

vLLM and SGLang are the two engines teams most often shortlist together. Both are open
source, both serve the OpenAI API, both do continuous batching over a paged KV cache,
and both now ship prefix caching, speculative decoding, structured outputs and
multi-GPU parallelism. On a plain chat benchmark they are often within noise of each
other, and the lead changes from release to release.

So "which is faster?" is the wrong question. The useful one is: **where do their designs
differ, and does my workload hit those differences?** This page goes layer by layer,
and every claim that can be simulated has a runnable model in this repo.

> Both projects move fast. The architecture described here is stable; specific flags and
> defaults change - check the release notes of the version you deploy.

---

## 1. Origins and design center

| | vLLM | SGLang |
|---|---|---|
| Started at | UC Berkeley Sky Computing Lab (2023) | LMSYS / UC Berkeley, Stanford (2023-24) |
| Founding paper | *Efficient Memory Management for LLM Serving with PagedAttention* | *SGLang: Efficient Execution of Structured Language Model Programs* |
| Founding problem | KV-cache **memory** waste limits batch size | **Programs** of many LLM calls waste work and are awkward to write |
| Unit of work | a request | a request, or a program of related requests |
| Center of gravity today | the default general-purpose serving engine; broadest model and hardware coverage | high-performance serving with a strong lead in prefix-heavy, agentic and large-MoE deployments |

The founding problems still explain most differences: vLLM's abstractions are built
around **memory blocks**, SGLang's around **token prefixes**.

## 2. Runtime architecture

Both split the server into processes so Python overhead doesn't stall the GPU.

```
vLLM (V1 engine)                              SGLang (SRT)
┌──────────────┐  ┌─────────────────────┐     ┌───────────┐  ┌────────────────────┐  ┌──────────────┐
│ API server   │─▶│ EngineCore          │     │ Tokenizer │─▶│ Scheduler          │─▶│ Detokenizer  │
│ tokenize,    │  │ scheduler + KV mgr  │     │ manager   │  │ radix cache,       │  │ manager      │
│ detokenize   │◀─│ + model executor(s) │     │ (HTTP)    │◀─│ policy, model      │◀─│              │
└──────────────┘  └─────────────────────┘     └───────────┘  │ worker(s)          │  └──────────────┘
                                                             └────────────────────┘
```

| Concern | vLLM | SGLang |
|---|---|---|
| Scheduler model | Unified token budget per step: prefill chunks and decodes share one batch (chunked prefill on by default in V1) | Separate prefill/decode batches with a configurable prefill chunk size; mixed chunking optional |
| CPU overhead hiding | Multiprocess engine core, persistent batch, CUDA graphs, `torch.compile` | **Overlap scheduler**: schedules batch *N+1* on CPU while the GPU runs batch *N* |
| Attention kernels | FlashAttention, FlashInfer, Triton and others selected per platform | FlashInfer, FlashAttention-3, Triton and others |
| Graph capture | CUDA graphs (piecewise, around attention) via `torch.compile` | CUDA graphs for decode; `torch.compile` optional |

Practical effect: at small batch sizes, CPU scheduling overhead is a visible share of
each step, which is exactly where overlap scheduling and graph capture matter. At large
batch sizes GPU time dominates and the engines converge.

## 3. KV cache and prefix reuse - the core difference

### vLLM: Automatic Prefix Caching (block hashes)

* KV lives in fixed-size blocks (PagedAttention).
* Each **full** block gets a hash of `(parent block hash, tokens in block)`, so equal
  hashes mean equal prefixes - a chained hash, like a Merkle list.
* A new request hashes its prompt block by block and reuses every leading cached block.
* Freed blocks stay in a cache and are evicted LRU only when memory is needed.
* The **partial last block is never shared**, so reuse is block-granular (16 tokens by
  default). Near-zero overhead, and on by default in the V1 engine.

### SGLang: RadixAttention (radix tree)

* Cached KV is indexed by a **radix tree** over token ids; edges hold token runs and split
  when two requests diverge mid-edge.
* Matching is **token-granular**, and the tree answers "how much of this request is
  cached?" cheaply for *every* waiting request.
* LRU eviction of leaves; nodes used by running requests are locked.

### What actually differs in practice

Run `ie simulate prefix` (code: [`concepts/prefix_caching.py`](../src/inference_engines/concepts/prefix_caching.py)).
128 requests from 8 interleaved prompt groups (1,000-token prefix + 60-token suffix each):

| KV capacity | Cache | Queue policy | Hit rate | Tokens prefilled |
|---|---|---|---|---|
| 3,000 tokens (pressure) | block-hash (vLLM) | FCFS | 18.3% | 110,880 |
| | radix (SGLang) | FCFS | 18.4% | 110,680 |
| | block-hash | **LPM** | 81.9% | 24,576 |
| | radix | **LPM** | 82.5% | 23,680 |
| 1,000,000 tokens (ample) | block-hash | either | 87.7% | 16,640 |
| | radix | either | 88.4% | 15,680 |

Two conclusions, which match what you see on real servers:

1. **The data structure barely matters.** Block hashing loses only the partial tail block
   (< 1 percentage point here). Both engines reuse shared prefixes well.
2. **The scheduling policy matters a lot under memory pressure.** With many interleaved
   long prefixes and a cache that can't hold them all, first-come-first-served evicts a
   prefix before the next request that needs it runs. SGLang's default **longest-prefix-
   match (LPM)** policy runs requests whose prefix is cached first, which is a 4.5×
   difference in prefill work in this simulation. The radix tree is what makes that
   policy cheap to evaluate, which is why it is SGLang's default and not vLLM's.

LPM's cost is fairness: a request with an uncached prefix can wait behind a stream of
cache hits (SGLang falls back to FCFS when the queue gets very long, because matching
every waiting request becomes expensive). vLLM's FCFS (or explicit priority) scheduling
is more predictable for tail latency on mixed traffic.

**When this matters for you:** many agents or tenants with *different* long system
prompts, few-shot templates, or RAG contexts sharing a GPU; multi-turn chat at high
concurrency; tree search / best-of-n over a shared trunk. When it doesn't: short
independent prompts, or one shared system prompt that always stays cached.

## 4. Programming model

| | vLLM | SGLang |
|---|---|---|
| Server API | OpenAI-compatible (chat, completions, embeddings, ...) | OpenAI-compatible + native `/generate` |
| Offline API | `LLM.generate` / `LLM.chat` - excellent for batch jobs | `sgl.Engine` offline engine |
| Program DSL | None (orchestrate in your code / an agent framework) | `@sgl.function` with `gen`, `select`, `fork`, `join`, regex/JSON constraints |
| Branching | Parallel sampling (`n>1`) shares the prompt's blocks | `fork` shares *all* KV up to the fork point, then branches run batched |

The DSL's advantage is that the runtime sees the whole program: it knows a `fork` is
coming, keeps the trunk cached, and batches branches together. You can get most of the
caching benefit from either engine through the OpenAI API alone; the DSL is about
expressing (and speeding up) the control flow. See
[`examples/sglang/agent_program.py`](../examples/sglang/agent_program.py).

## 5. Structured output

Both compile JSON schemas / regexes / grammars into token masks and apply them on the GPU
batch every step; both use **XGrammar** as a default backend today (vLLM also offers
`guidance` and `outlines` backends). SGLang introduced **jump-forward decoding** - when the
grammar admits exactly one continuation, append it without a forward pass.

Model: [`concepts/constrained.py`](../src/inference_engines/concepts/constrained.py)
(`ie simulate constrained`: 93 → 16 model calls for the same JSON object).

Reality check: jump-forward helps most for verbose, schema-heavy outputs (lots of fixed
keys); for short outputs mask computation overhead and tokenization boundaries dominate,
and both engines are fast. The OpenAI `response_format={"type": "json_schema", ...}`
request is portable between the two -
[`examples/sglang/structured_output.py`](../examples/sglang/structured_output.py) runs on both.

## 6. Speculative decoding

| | vLLM | SGLang |
|---|---|---|
| Methods | EAGLE / EAGLE-3, MTP heads (e.g. DeepSeek), n-gram (prompt lookup), draft models | EAGLE / EAGLE-3, MTP, draft models, n-gram |
| Interaction with batching | Integrated into the V1 scheduler | Integrated, with tree-based drafting (`--speculative-eagle-topk`) |

Both preserve the target distribution exactly (proved statistically in
[`tests/test_speculative.py`](../tests/test_speculative.py)). Speculation pays most at
**low batch sizes** (latency-bound) and fades at high batch sizes, where the GPU is
already busy; `ie simulate speculative` shows how acceptance rate drives the gain.

## 7. Parallelism and large models

| | vLLM | SGLang |
|---|---|---|
| Tensor / pipeline parallel | Yes / Yes | Yes / Yes |
| Data parallel | Yes, incl. DP attention for MLA models | Yes; **DP attention** (pioneered for DeepSeek MLA) |
| Expert parallel (MoE) | Yes, with optimized all-to-all kernels | Yes, with optimized all-to-all kernels |
| Prefill/decode disaggregation | Yes (KV-transfer connectors, e.g. NIXL; llm-d) | Yes (Mooncake / NIXL transfer backends) |
| Multi-replica routing | vLLM production-stack router, llm-d | `sgl-router` with cache-aware load balancing |

Both run frontier MoE models (DeepSeek, Qwen-MoE, Llama 4 ...) at multi-node scale.
SGLang has been notably early on large-scale DeepSeek-style deployments (DP attention +
EP + PD disaggregation); vLLM has caught up quickly. Cache-aware routing - send a
request to the replica that already holds its prefix - is the cluster-level version of
section 3, and both ecosystems now have it.

## 8. Models, hardware and ecosystem

| | vLLM | SGLang |
|---|---|---|
| Model coverage | Broadest (hundreds of architectures, day-0 support common) | Broad, focused on popular families; day-0 for many major releases |
| Hardware | NVIDIA, AMD, Intel GPUs, Google TPU, AWS Neuron, Gaudi, CPU, via a plugin system | NVIDIA, AMD, plus growing support for other accelerators |
| LoRA | Multi-LoRA serving | Multi-LoRA serving |
| Governance / community | PyTorch Foundation project; largest contributor base | LMSYS-led, very active; widely used in large production deployments |
| Learning curve | Lower; good defaults | Slightly steeper, especially the DSL and advanced flags |

## 9. Head-to-head summary

| You need... | Lean | Why |
|---|---|---|
| A general production chat/completions API | **vLLM** | Mature defaults, broadest models/hardware, biggest community |
| Offline batch inference in Python | **vLLM** | `LLM.chat` over thousands of prompts is simple and fast |
| Non-NVIDIA accelerators (TPU, Neuron, Gaudi, Intel) | **vLLM** | Plugin ecosystem |
| Agents / multi-call programs with long, varied prefixes | **SGLang** | LPM scheduling + radix cache keeps prefixes hot under pressure |
| Branching: tree search, best-of-n, parallel tool calls | **SGLang** | `fork` over shared KV, batched branches |
| Heavy structured output | **SGLang** (slight) | Jump-forward; both are good |
| Very large MoE at multi-node scale | either - benchmark | Both strong; SGLang was early, vLLM caught up |
| Low concurrency, latency-bound | either + speculative decoding | Both have EAGLE-3 / MTP |

## 10. Benchmarking them fairly

[`examples/compare/vllm_vs_sglang.py`](../examples/compare/vllm_vs_sglang.py) runs three
workloads that isolate the differences above - `chat` (raw batching), `shared` (one long
shared prefix) and `groups` (many interleaved prefixes) - against both servers at several
concurrency levels:

```bash
docker compose -f deploy/docker-compose.yml --profile vllm up -d      # then stop it and
docker compose -f deploy/docker-compose.yml --profile sglang up -d    # run the other, same GPU
python examples/compare/vllm_vs_sglang.py --concurrency 8 32 128
```

Checklist (most "X is 2× faster than Y" posts violate at least one):

1. Same model revision, dtype, quantization, and max context length.
2. Prefix caching **on for both** (or off for both), stated explicitly.
3. Same chat template and sampling params; fixed `max_tokens`, ignore EOS if comparing decode.
4. Warm-up requests first (graph capture, JIT, cold caches).
5. One engine per GPU at a time; idle GPU otherwise.
6. Sweep concurrency and report **p50/p99 TTFT, TPOT, and throughput**, not one number.
7. Use *your* prompt/output length distribution - see [benchmarking.md](benchmarking.md).

## 11. The migration angle

Because both speak the same API, switching is cheap and **A/B testing is cheap**: run
both behind your gateway, mirror a slice of traffic, compare p99 TTFT and cost per
million tokens on your real workload. A common path is vLLM first (fastest to
production), then trying SGLang when workloads become agentic or prefix-heavy - same
model, same application code ([`examples/migration/app.py`](../examples/migration/app.py)).
