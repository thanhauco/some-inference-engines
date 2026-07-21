# Benchmarking engines properly

## Metrics

| Metric | Meaning | Who feels it |
|---|---|---|
| **TTFT** | time to first token = queueing + prefill | chat users ("is it thinking?") |
| **TPOT / ITL** | time per output token after the first | streaming readers, agents chaining calls |
| **E2E latency** | TTFT + (n-1) × TPOT | batch jobs, tool calls |
| **Throughput** | output tokens/s across all requests | your GPU bill |
| **Goodput** | requests/s that met *both* TTFT and TPOT SLOs | everyone |

Always report percentiles (p50/p90/p99). p99 TTFT moves first when a server saturates.

## Closed vs open loop

* **Closed loop** (`--concurrency N`): N users, each waits for its answer before asking again.
* **Open loop** (`--rate R`): Poisson arrivals at R req/s whatever the server does. Shows queueing that closed loops hide.

## Sweep, don't sample

```bash
ie bench vllm sglang --concurrency 1 4 16 64 128 --requests 256 --prompts my_prompts.txt
```

Throughput rises with concurrency until the batch is full or the KV cache is - then TTFT
explodes. That *knee* is the capacity of an engine for your workload. One number from
one concurrency level hides it.

## Your workload, not theirs

Prompt and output lengths change everything: long prompts stress prefill (TTFT), long
outputs stress decode (TPOT) and KV memory, shared prefixes reward prefix caching. Feed
`--prompts` with real (anonymized) prompts.

## Use the engines' own tools too

`vllm bench serve` and `python -m sglang.bench_serving` generate standardized load with
known datasets, which makes your numbers comparable to published ones. This repo's
`ie bench` complements them with one harness that runs identically against all four
engines, including Ollama and `trtllm-serve`.
