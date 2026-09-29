# You may not stay with the same engine forever

```
  Ollama ───────────▶ vLLM ───────────────▶ SGLang ──────────────────▶ TensorRT-LLM
  experimenting       real users arrive      workflows get complex      latency / GPU cost critical
```

The model can stay the same. Changing the engine underneath can completely change how
that model performs in production. That is why inference engineering is becoming a
discipline of its own.

## Make switching cheap from day one

1. **Code against the OpenAI API, not an engine.** All four implement
   `/v1/chat/completions`. [`examples/migration/app.py`](../examples/migration/app.py)
   runs unchanged on every engine; only `ENGINE=` changes.
2. **Keep the model family constant.** This repo's defaults use Llama 3.1 8B Instruct
   everywhere (`llama3.1:8b` on Ollama, `meta-llama/Llama-3.1-8B-Instruct` elsewhere).
   Note that Ollama's default tags are 4-bit quantized, so outputs can differ slightly
   from a BF16 server - evaluate quality when you migrate, not only speed.
3. **Keep config out of code.** `IE_<ENGINE>_BASE_URL`, `IE_<ENGINE>_MODEL`,
   `IE_<ENGINE>_API_KEY` ([`config.py`](../src/inference_engines/config.py)).
4. **Keep a benchmark of your own traffic.** `ie bench` with your prompts file; re-run it
   at every migration and every engine upgrade.

## Stage by stage

### 1. Experimenting → Ollama
Pick a model, iterate on prompts, build the first version of the app on a laptop.
Exit signal: more than a handful of concurrent users, or you need a shared GPU server.

### 2. Real users → vLLM
Same model in BF16/FP8 on a GPU server, continuous batching, prefix caching,
Prometheus metrics, Kubernetes ([`deploy/k8s/vllm.yaml`](../deploy/k8s/vllm.yaml)).
Exit signal: requests become multi-step programs, or prefix reuse under load is poor.

### 3. Complex workflows → SGLang
Agents, tool calls, branching, schema-constrained outputs, many long prefixes. Try it as
a drop-in OpenAI server first; adopt the frontend language where control flow benefits.
A/B against vLLM with [`examples/compare/vllm_vs_sglang.py`](../examples/compare/vllm_vs_sglang.py).
Exit signal: latency SLOs or GPU spend dominate the roadmap.

### 4. Latency / economics critical → TensorRT-LLM
FP8/FP4, fused kernels, per-GPU tuned engines. Build engines in CI, pin versions, and
keep the previous engine deployable for rollback.

## Rollout checklist for any switch

- [ ] Quality eval on your own test set (tokenizer/chat template and quantization differences show up here)
- [ ] `ie bench` sweep at expected and 2× peak concurrency; compare p99 TTFT, TPOT, goodput
- [ ] Shadow or canary traffic behind the same gateway
- [ ] Dashboards: queue depth, KV-cache usage, TTFT/TPOT percentiles, error rate
- [ ] Rollback = change one base URL
