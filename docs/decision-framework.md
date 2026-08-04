# Which engine should you choose, and when?

> If you are choosing an inference engine based on benchmarks alone, you are probably
> choosing it the wrong way.

Benchmarks answer "which engine is fastest *for that benchmark's* model, prompt lengths,
concurrency and GPU". The inference layer also decides your developer experience, your
operational load, your hardware options and your cost per token. So decide on the
*shape of your problem* first, then benchmark the shortlist on your own workload.

## The framework

| Engine | One line | Choose it when |
|---|---|---|
| **Ollama** | simplest local experience | you want the fastest path to local experimentation and production-scale concurrency is not yet your problem |
| **vLLM** | scalable production serving | you are building a production API and care about concurrency, throughput and GPU utilization without going deep into hardware-specific tuning |
| **SGLang** | programmable and agentic inference | your application starts behaving more like an inference program than a simple model endpoint |
| **TensorRT-LLM** | maximum NVIDIA performance | every millisecond matters and you are willing to trade simplicity for NVIDIA-specific optimization |

## Five questions

1. **Where does it run?** Laptop/CPU/Apple Silicon → Ollama. No NVIDIA GPU → TensorRT-LLM is out. AMD or other accelerators → vLLM first.
2. **How many concurrent users?** 1-8 → anything works, Ollama is simplest. Dozens to thousands → vLLM, SGLang or TensorRT-LLM (continuous batching + paged KV).
3. **What is a "request"?** Prompt → response → vLLM. Multi-step programs, tools, branches, long shared prefixes, schemas → SGLang.
4. **What is a millisecond worth?** If p99 latency or tokens-per-dollar at scale directly drives revenue or cost → TensorRT-LLM (or carefully tuned vLLM/SGLang).
5. **Who will operate it?** No inference specialists → Ollama/vLLM. A dedicated team → anything, including TensorRT-LLM engine builds.

## As code

[`decision.py`](../src/inference_engines/decision.py) scores each engine with explicit,
explained points and hard constraints:

```console
$ ie recommend --hardware nvidia-datacenter --concurrency 64 --workload agentic
1. sglang        score   9.3
      +2.0  high-performance GPU runtime
      +5.0  RadixAttention reuses shared prefixes across calls; fork/select/gen programs
      +2.3  continuous batching with cache-aware scheduling
2. vllm          score   8.0
...
=> sglang
```

The weights are opinions - deliberately visible so your team can argue with them and
adjust them, instead of arguing with a benchmark chart.

## Anti-patterns

* **Picking TensorRT-LLM for a prototype.** Engine builds slow iteration when the model and prompts still change weekly.
* **Shipping Ollama to production traffic.** It works for a handful of users; it is not built for hundreds of concurrent requests.
* **Choosing on a single throughput number.** Look at p99 TTFT at your concurrency, and at goodput (requests meeting your SLOs).
* **Assuming the choice is permanent.** It isn't - see [migration-path.md](migration-path.md).
