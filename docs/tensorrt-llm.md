# TensorRT-LLM - maximum NVIDIA performance

> **Choose it when** every millisecond matters and you are willing to trade simplicity
> for deeper NVIDIA-specific optimization.

## Compile, then serve

vLLM and SGLang run models mostly "as written" with optimized kernels. TensorRT-LLM
treats the model as a program to **compile** for one GPU:

```
HF / ONNX weights ──convert──▶ TRT-LLM checkpoint ──trtllm-build──▶ engine ("plan")
                    (layout, TP shards,                 (fused kernels, chosen
                     quantization)                       GEMM tactics, max shapes)
```

The engine is specific to the GPU architecture (an H100 engine won't run on an L40S) and
to the max batch / input / sequence lengths baked in at build time. The modern
**LLM API** and **`trtllm-serve`** hide most of this - give it a Hugging Face id and it
builds or loads an optimized runtime - but the trade-off is the same.

Scripts: [`examples/tensorrt_llm/build_engine.sh`](../examples/tensorrt_llm/build_engine.sh),
[`serve.sh`](../examples/tensorrt_llm/serve.sh),
[`llm_api_fp8.py`](../examples/tensorrt_llm/llm_api_fp8.py).

## Where the speed comes from

### Kernel fusion and layer fusion

`gelu(x @ W + b) + residual` is four kernels in eager mode, each round-tripping
activations through HBM. Fused, it is one kernel with one read and one write. Q/K/V
projections become a single GEMM. Code:
[`concepts/fusion.py`](../src/inference_engines/concepts/fusion.py) - `ie simulate fusion`
shows 4× less activation traffic and 96 fewer kernel launches per forward pass for one
fused epilogue in an 8B model.

### Quantization

Decode time ≈ bytes of weights / memory bandwidth, so fewer bits = faster tokens and
more room for KV cache.

| Format | Bytes vs BF16 | Notes |
|---|---|---|
| FP8 (W8A8) | 0.5× | Hopper/Ada/Blackwell tensor cores; usually near-lossless |
| INT8 SmoothQuant | 0.5× | Activation outliers migrated into weights |
| INT4 AWQ / GPTQ (W4A16) | ~0.27× | Group-wise scales; best for memory-bound decode |
| NVFP4 | ~0.28× | Blackwell |

Code: [`concepts/quantization.py`](../src/inference_engines/concepts/quantization.py) -
emulates INT8, INT4 group-wise and FP8 E4M3 rounding and compares error vs size
(`ie simulate quantization`).

### Runtime

In-flight batching (TensorRT-LLM's continuous batching), paged KV cache with optional
FP8/INT8 KV, CUDA graphs, multi-stream execution, speculative decoding, and
multi-GPU/multi-node tensor, pipeline and expert parallelism.

## Strengths and limits

| Pros | Cons |
|---|---|
| Best latency and throughput on NVIDIA GPUs | NVIDIA only |
| Deep quantization support (FP8, INT4, FP4) | Build / engine process adds complexity |
| Production-grade, backs NVIDIA Dynamo / Triton | Newest models may land later |
| Best GPU economics at scale | Less flexible for agentic workflows |

## Operational advice

* Pin the container tag; engines built with one TensorRT-LLM version are not portable to another.
* Build engines in CI per GPU type and store them as artifacts; never build on the serving path.
* Size `max_batch_size` / `max_num_tokens` from a load test, not guesses - they are fixed after the build.
