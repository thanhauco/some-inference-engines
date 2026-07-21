"""Async load generator for OpenAI-compatible engines.

Two ways to apply load, and they answer different questions:

* **Closed loop** (``concurrency=N``): N virtual users, each sends its next request when
  the previous finishes. Answers "how does the engine behave with N users?".
* **Open loop** (``request_rate=R``): Poisson arrivals at R req/s regardless of how the
  server copes. Answers "what happens at this traffic level?" - and exposes queueing,
  which closed-loop tests hide.

Run a *sweep* over concurrency to find the knee where throughput stops growing and TTFT
explodes; that knee, not a single number, is the engine's capacity for your workload.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..client import AsyncInferenceClient, Message
from .metrics import RequestRecord, Summary, summarize


@dataclass
class LoadConfig:
    num_requests: int = 100
    concurrency: int = 8
    request_rate: float | None = None
    """Requests/second for open-loop Poisson arrivals; ``None`` = closed loop."""
    max_tokens: int = 256
    temperature: float = 0.0
    model: str | None = None
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    engine: str
    config: LoadConfig
    records: list[RequestRecord]
    summary: Summary


async def _one(
    client: AsyncInferenceClient, messages: list[Message], cfg: LoadConfig, t0: float
) -> RequestRecord:
    start = time.perf_counter()
    try:
        completion = await client.timed_stream(
            messages,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            **cfg.extra,
        )
    except Exception as exc:  # noqa: BLE001 - a benchmark records failures, it doesn't stop
        return RequestRecord(start - t0, time.perf_counter() - start, None, 0, repr(exc))
    tokens = completion.completion_tokens or len(completion.chunk_times_s)
    return RequestRecord(start - t0, completion.latency_s, completion.ttft_s, tokens)


async def run_load(
    client: AsyncInferenceClient, prompts: Sequence[list[Message]], cfg: LoadConfig
) -> BenchmarkResult:
    if not prompts:
        raise ValueError("need at least one prompt")
    rng = random.Random(cfg.seed)
    work = [prompts[i % len(prompts)] for i in range(cfg.num_requests)]
    sem = asyncio.Semaphore(cfg.concurrency)
    t0 = time.perf_counter()

    async def gated(messages: list[Message]) -> RequestRecord:
        async with sem:
            return await _one(client, messages, cfg, t0)

    tasks = []
    for messages in work:
        tasks.append(asyncio.create_task(gated(messages)))
        if cfg.request_rate:
            await asyncio.sleep(rng.expovariate(cfg.request_rate))
    records = list(await asyncio.gather(*tasks))
    duration = time.perf_counter() - t0
    return BenchmarkResult(client.profile.engine.value, cfg, records, summarize(records, duration))


async def sweep(
    client: AsyncInferenceClient,
    prompts: Sequence[list[Message]],
    concurrencies: Sequence[int] = (1, 4, 16, 64),
    requests_per_level: int = 64,
    **cfg_kwargs,
) -> list[BenchmarkResult]:
    results = []
    for c in concurrencies:
        cfg = LoadConfig(num_requests=max(requests_per_level, c), concurrency=c, **cfg_kwargs)
        results.append(await run_load(client, prompts, cfg))
    return results


DEFAULT_PROMPTS: list[list[Message]] = [
    [{"role": "user", "content": "Explain continuous batching in two sentences."}],
    [{"role": "user", "content": "Write a haiku about GPUs."}],
    [
        {"role": "system", "content": "You are a terse SRE assistant."},
        {"role": "user", "content": "List five causes of high p99 latency in an LLM API."},
    ],
    [{"role": "user", "content": "Summarize the trade-offs of 4-bit quantization."}],
    [{"role": "user", "content": "Return a JSON object with keys city and country for Paris."}],
]
