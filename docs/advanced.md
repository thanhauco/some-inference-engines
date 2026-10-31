# Advanced topics: what decides performance after you picked an engine

Choosing the engine is step one. These five topics are where production deployments
actually win or lose latency and money, and each has a GPU-free model in
[`src/inference_engines/concepts`](../src/inference_engines/concepts) you can run and tune.

| Topic | Command | Applies to |
|---|---|---|
| Prefill/decode interference, chunked prefill, PD disaggregation | `ie simulate interference` | vLLM, SGLang, TensorRT-LLM |
| Cache-aware routing across replicas | `ie simulate routing` | any multi-replica deployment |
| Tensor parallelism vs replicas | `ie simulate parallelism` | multi-GPU serving |
| GPU/CPU layer offload and cold starts | `ie simulate offload` | Ollama (llama.cpp) |
| Capacity and cost per million tokens | `ie simulate capacity` | everything |

---

## 1. Prefill and decode fight over the same GPU

Prefill is compute-bound (hundreds of ms for a long prompt); a decode step is ~17 ms. On
one GPU, a single 32k-token prompt arriving mid-stream freezes every sequence that is
decoding. [`prefill_decode.py`](../src/inference_engines/concepts/prefill_decode.py)
streams 32 sequences while six 32k-token prompts arrive:

| mode | ITL p50 | ITL max | long-prompt TTFT |
|---|---|---|---|
| monolithic prefill | 16.8 ms | **657 ms** | 665 ms |
| chunked prefill (2,048 tokens/step) | 16.8 ms | **58 ms** | 917 ms |
| disaggregated (separate prefill GPU) | 16.8 ms | **16.8 ms** | 721 ms |

* **Chunked prefill** bounds the stall with a per-step token budget. It is on by default
  in vLLM's V1 engine, configurable in SGLang (`--chunked-prefill-size`) and available in
  TensorRT-LLM (chunked context). Smaller chunks = smoother streams, slower TTFT for long
  prompts; tune it against your TPOT SLO.
* **Prefill/decode disaggregation** runs the two phases on separate pools and ships the KV
  cache between them (SGLang PD mode, vLLM with KV-transfer connectors, NVIDIA Dynamo).
  Decode latency becomes independent of prompt traffic and each pool is sized on its own;
  the price is an extra pool, a fast interconnect and a more complex deployment. Worth
  it when prompts are long and bursty (RAG, agents with big tool outputs).

## 2. Prefix caching only works if the request lands on the right GPU

Section 3 of [vllm-vs-sglang.md](vllm-vs-sglang.md) covers prefix reuse on *one* GPU.
Behind a load balancer the question becomes which replica holds the prefix.
[`routing.py`](../src/inference_engines/concepts/routing.py) sends 16 agents × 24 requests
to 4 replicas:

| routing policy | cache hit rate | busiest replica / mean |
|---|---|---|
| round robin | **16.5%** | 1.00 |
| consistent hash on the prompt head | 68.1% | 1.25 |
| prefix affinity + balance guard | **67.8%** | 1.01 |

Round robin makes every replica compute every prefix. Both cache-aware policies recover
4× the hit rate; prefix affinity also keeps load even, because it falls back to the
least-loaded replica when the preferred one is overloaded. This is what `sgl-router`'s
cache-aware policy, the vLLM production-stack router and the Kubernetes Gateway API
inference extension (llm-d) implement. If you run more than one replica, check what your
gateway does - a plain L7 round robin silently throws away most of the cache.

## 3. Tensor parallelism buys latency; replicas buy throughput

[`parallelism.py`](../src/inference_engines/concepts/parallelism.py) splits 4 GPUs every
possible way for an 8B model:

| layout (NVLink) | TPOT | total tok/s |
|---|---|---|
| 4 replicas × TP=1 | 16.8 ms | **7,619** |
| 2 replicas × TP=2 | 9.4 ms | 6,838 |
| 1 replica × TP=4 | **5.2 ms** | 6,202 |

TP shards the weights, so each step reads 1/n of them per GPU - but every layer adds two
all-reduce exchanges that do not shrink. With a slow interconnect the same model says
TP=2 is *slower* than TP=1 (27.6 ms vs 16.8 ms across nodes), which is why TP across
nodes is rare and TP within an NVLink box is routine.

Rules of thumb:

1. Model fits on one GPU and you care about cost → replicas.
2. Model fits but TPOT misses the SLO → TP within one node.
3. Model does not fit (70B in BF16 on 80 GB) → TP is mandatory; then consider FP8 to get
   back to one GPU per replica.

## 4. Ollama: offload is a cliff, and idle models unload

[`offload.py`](../src/inference_engines/concepts/offload.py) models llama.cpp's layer
split for Llama 3.1 8B `q4_K_M`:

| VRAM | layers on GPU | tok/s (illustrative timings) |
|---|---|---|
| 0 GiB | 0 / 32 | 7.8 |
| 4 GiB | 19 / 32 | 16.3 |
| 6 GiB | 31 / 32 | 51.3 |
| 8 GiB | 32 / 32 | 62.5 |

Every token passes through every layer, so the CPU layers dominate: with 60% of the
layers on the GPU you get 26% of the speed. Practical consequences:

* Pick the quantization that fits **entirely** in VRAM (`recommend_quant`), including the
  context window - a longer `num_ctx` can push layers back to the CPU.
* `ollama ps` shows the CPU/GPU split; anything other than `100% GPU` is the first thing
  to fix when Ollama feels slow.
* Models unload after `OLLAMA_KEEP_ALIVE` (5 minutes by default). With sparse traffic
  nearly every request pays a cold start (~2.4 s for this model from NVMe). Set
  `OLLAMA_KEEP_ALIVE=-1` for anything that behaves like a service.

## 5. Economics: tokens per second become GPUs and dollars

[`capacity.py`](../src/inference_engines/concepts/capacity.py) turns *your measured*
throughput into a fleet size and a price:

```
$ / 1M tokens = GPU $/hour ÷ (tok/s × 3600 × utilization) × 1e6
GPUs          = peak tok/s ÷ (tok/s per GPU × target utilization)
break-even    = migration cost ÷ monthly GPU savings
```

Two things follow directly from the formulas:

* **Utilization is worth as much as speed.** A GPU at 30% utilization costs 3.3× more per
  token than at 100%. Continuous batching, prefix caching and autoscaling raise
  utilization; that is often a bigger win than a faster kernel.
* **A migration is an investment with a payback period.** If moving engines yields 80% more
  throughput per GPU at a 50k tok/s peak, 36 GPUs become 20 and a $60k engineering effort
  pays back in ~2 months (with the placeholder numbers in `ie simulate capacity` - plug in
  your own from `ie bench`). At a 2k tok/s peak the same migration never pays back. This
  is the quantitative version of "explore TensorRT-LLM when GPU economics become critical".

---

## How the pieces fit together

```
                         ┌──────────── gateway: cache-aware routing (2) ────────────┐
 requests ──▶ router ──▶ │ replica 1   replica 2   …   replica N   (replicas vs TP: 3)│
                         └─────────────────────────────────────────────────────────┘
 inside a replica:  scheduler ─ chunked prefill / PD split (1) ─ prefix cache ─ paged KV
 around all of it:  capacity plan and cost per token (5)
```

Start with the engine that fits the shape of your problem; then these are the knobs, in
roughly the order they pay off: prefix caching and routing, chunked prefill, quantization,
parallelism layout, and finally disaggregation.
