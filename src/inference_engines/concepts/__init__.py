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
=====================  ==============================  ================
"""
