# SGLang - programmable and agentic inference

> **Choose it when** your application starts behaving more like an *inference program*
> than a simple model endpoint.

## Prompt → response vs. inference programs

A chatbot sends one prompt and gets one answer. An agent does something like:

```
classify ticket ─▶ branch ─▶ call tool ─▶ draft 3 replies in parallel ─▶ pick best ─▶ JSON
       └──────────────── every step re-sends the same long prefix ────────────────┘
```

Served naively, every step re-prefills the system prompt, tool definitions and history,
parallel branches each recompute the shared trunk, and structured outputs need a parse-
and-retry loop. SGLang ("Structured Generation Language") attacks each of these.

## The pieces

### Frontend language

Python-embedded primitives: `gen`, `select`, `fork`/`join`, roles, and constraints
(`regex=`, JSON schema). See [`examples/sglang/agent_program.py`](../examples/sglang/agent_program.py):

```python
@sgl.function
def handle_ticket(s, ticket):
    s += sgl.user(ticket)
    s += sgl.assistant(sgl.select("team", choices=["billing", "performance", "bug", "other"]))
    drafts = s.fork(3)                       # 3 branches share all KV computed so far
    for d in drafts:
        d += sgl.assistant(sgl.gen("reply", max_tokens=160))
```

The same capabilities (structured output, tool calls) are also reachable through the
OpenAI-compatible API, so you don't *have* to adopt the DSL.

### RadixAttention

Finished requests leave their KV cache in a **radix tree** keyed by tokens. A new
request reuses the longest cached prefix and only prefills the rest; LRU leaf eviction
keeps hot prefixes (system prompts) resident. Code:
[`concepts/radix_cache.py`](../src/inference_engines/concepts/radix_cache.py).
`ie simulate radix`: 8 agent sessions × 5 turns with a 1.5k-token system prompt prefill
**~12× fewer tokens**.

### Cache-aware scheduling

The default policy serves waiting requests with the **longest cached prefix** first, so
a cached prefix is used by everyone waiting for it before it can be evicted. Under
memory pressure this is the difference between ~18% and ~82% cache hit rate in
`ie simulate prefix` - see [vllm-vs-sglang.md](vllm-vs-sglang.md).

### Constrained decoding with jump-forward

JSON schemas / regexes compile to a grammar automaton; invalid tokens are masked each
step so output is valid by construction. When the automaton has only one legal
continuation (`{"name": "`), those tokens are appended **without a forward pass**. Code:
[`concepts/constrained.py`](../src/inference_engines/concepts/constrained.py) - even a
random "model" emits valid JSON, with ~6× fewer model calls using jump-forward.

### Speculative decoding

EAGLE-style drafting: a light draft head proposes several tokens; the target model
verifies them in one pass; accepted tokens are free. Output distribution is unchanged.
Code: [`concepts/speculative.py`](../src/inference_engines/concepts/speculative.py) -
including the test that proves the output distribution matches the target exactly.

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path <eagle3-draft> \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

## Strengths and limits

| Pros | Cons |
|---|---|
| Flexible programmable interface | Steeper learning curve |
| High performance for complex, prefix-heavy workloads | Ecosystem younger than vLLM's |
| Parallel tool calls / branching built in | Primarily optimized for NVIDIA (AMD supported) |
| Fast structured outputs | |
