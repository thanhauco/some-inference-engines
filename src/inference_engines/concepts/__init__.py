"""Pure-Python models of what happens *inside* each engine.

Nothing here needs a GPU. Each module isolates one idea from the diagram so you can run
it, change a parameter and watch the effect:

=====================  ==============================  ================
module                 idea                            engine
=====================  ==============================  ================
``kv_cache``           PagedAttention block manager    vLLM
``batching``           static vs continuous batching   vLLM
``radix_cache``        RadixAttention prefix reuse     SGLang
``speculative``        draft / verify decoding         SGLang (+others)
``constrained``        FSM-constrained JSON decoding   SGLang
``quantization``       INT8 / INT4 / FP8 weights       TensorRT-LLM
``fusion``             kernel and layer fusion         TensorRT-LLM
``memory``             will this model fit?            Ollama (+all)
``offload``            GPU/CPU layer split, cold start Ollama
``prefix_caching``     block-hash vs radix + LPM       vLLM vs SGLang
``prefill_decode``     chunked prefill, PD disaggreg.  vLLM, SGLang, TRT
``routing``            cache-aware load balancing      cluster level
``parallelism``        tensor parallel vs replicas     multi-GPU
``capacity``           GPUs and $ per million tokens   economics
=====================  ==============================  ================
"""
