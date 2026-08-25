"""Head-to-head: vLLM vs SGLang on the same model, same GPU, three workload shapes.

Each workload isolates one architectural difference (see docs/vllm-vs-sglang.md):

  chat        independent short prompts         -> raw batching/decoding efficiency
  shared      one long system prompt, many users -> prefix caching (APC vs RadixAttention)
  groups      many interleaved long prefixes     -> cache-aware scheduling under memory pressure

Start both servers with the same model, dtype and context length, e.g. via
`docker compose --profile vllm --profile sglang up` (deploy/docker-compose.yml), then:

    python examples/compare/vllm_vs_sglang.py --concurrency 8 32 128
    python examples/compare/vllm_vs_sglang.py --simulate-only     # no GPU needed

Fairness checklist: same model revision and dtype; same max context; prefix caching on
for both (vLLM: --enable-prefix-caching, SGLang: on by default); warm both up; same
sampling params; run each level more than once and on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import random

from inference_engines.benchmark.loadgen import LoadConfig, run_load
from inference_engines.benchmark.report import to_markdown
from inference_engines.client import AsyncInferenceClient
from inference_engines.concepts.prefix_caching import head_to_head

TOPICS = [
    "KV cache",
    "tensor parallelism",
    "FP8",
    "speculative decoding",
    "LoRA",
    "RAG",
    "chunked prefill",
    "CUDA graphs",
    "tokenizers",
    "beam search",
]


def _filler(seed: int, words: int) -> str:
    rng = random.Random(seed)
    vocab = "latency throughput batch token cache memory kernel request queue model".split()
    return " ".join(rng.choice(vocab) for _ in range(words))


def chat_workload(n: int = 64):
    return [
        [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} briefly. (#{i})"}]
        for i in range(n)
    ]


def shared_prefix_workload(n: int = 64, prefix_words: int = 1500):
    system = "You are an inference expert. Reference manual:\n" + _filler(0, prefix_words)
    return [
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question {i}: what about {TOPICS[i % len(TOPICS)]}?"},
        ]
        for i in range(n)
    ]


def grouped_prefix_workload(groups: int = 12, per_group: int = 8, prefix_words: int = 1500):
    prompts = []
    for g in range(groups):
        system = f"Agent #{g}. Tool manual:\n" + _filler(g + 1, prefix_words)
        prompts += [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task {i} for agent {g}."},
            ]
            for i in range(per_group)
        ]
    random.Random(0).shuffle(prompts)
    return prompts


WORKLOADS = {
    "chat": chat_workload,
    "shared": shared_prefix_workload,
    "groups": grouped_prefix_workload,
}


def simulate() -> None:
    print("Simulated prefix reuse (see inference_engines/concepts/prefix_caching.py):\n")
    for capacity in (3000, 1_000_000):
        print(f"KV capacity {capacity:,} tokens")
        for r in head_to_head(capacity_tokens=capacity):
            print(
                f"  {r.cache:<18} {r.policy:<5} hit rate {r.hit_rate:6.1%}  "
                f"prefilled {r.prefilled_tokens:,}"
            )
    print()


async def live(concurrency: list[int], requests: int, max_tokens: int) -> None:
    for name, make in WORKLOADS.items():
        prompts = make()
        results = []
        for engine in ("vllm", "sglang"):
            async with AsyncInferenceClient(engine) as client:
                # Warm-up so neither engine pays compile/graph-capture or cold-cache costs.
                await run_load(
                    client, prompts[:4], LoadConfig(num_requests=4, concurrency=4, max_tokens=16)
                )
                for c in concurrency:
                    cfg = LoadConfig(
                        num_requests=max(requests, c), concurrency=c, max_tokens=max_tokens
                    )
                    results.append(await run_load(client, prompts, cfg))
        print(f"\n### workload: {name}\n")
        print(to_markdown(results))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    simulate()
    if not args.simulate_only:
        asyncio.run(live(args.concurrency, args.requests, args.max_tokens))


if __name__ == "__main__":
    main()
