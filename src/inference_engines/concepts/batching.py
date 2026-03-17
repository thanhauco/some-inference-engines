"""Static batching vs continuous batching, the throughput idea behind vLLM.

Decoding is memory-bound: each step streams all model weights from HBM to produce *one*
token per sequence, so a step over 32 sequences costs barely more than a step over 1.
Batching is how a server turns that into throughput. *How* you batch matters:

**Static batching** collects a batch, runs it until the *longest* request finishes, then
starts the next. Short requests finish early but their slots sit idle (padding), and new
arrivals wait for the whole batch. Without paging, each slot also reserves KV memory for
the maximum sequence length, which caps the batch size.

**Continuous (iteration-level) batching** schedules every decode step: finished sequences
leave immediately and waiting ones join the very next step, as long as the paged KV cache
has blocks for them. If the cache runs out mid-generation, the newest sequence is
preempted (its blocks freed, recomputed later) - exactly vLLM's default policy.

Both schedulers below run on the same deterministic cost model so their numbers are
directly comparable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from .kv_cache import PagedKVCache


@dataclass(frozen=True)
class Request:
    id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CostModel:
    """Time for one forward step of the model, in milliseconds.

    ``decode_base_ms`` is the cost of streaming the weights once (paid per step regardless
    of batch size); ``decode_per_seq_ms`` is the marginal cost of one more sequence;
    ``prefill_per_token_ms`` is the compute-bound cost of processing prompt tokens.
    The defaults are in the ballpark of an 8B model on one datacenter GPU.
    """

    decode_base_ms: float = 12.0
    decode_per_seq_ms: float = 0.15
    prefill_per_token_ms: float = 0.02

    def step_ms(self, batch_size: int, prefill_tokens: int = 0) -> float:
        return (
            self.decode_base_ms
            + self.decode_per_seq_ms * batch_size
            + self.prefill_per_token_ms * prefill_tokens
        )


@dataclass
class _Running:
    request: Request
    generated: int = 0
    first_token_ms: float | None = None
    finish_ms: float | None = None


@dataclass
class RequestResult:
    id: int
    arrival_ms: float
    first_token_ms: float
    finish_ms: float
    output_tokens: int

    @property
    def ttft_ms(self) -> float:
        return self.first_token_ms - self.arrival_ms

    @property
    def latency_ms(self) -> float:
        return self.finish_ms - self.arrival_ms


@dataclass
class SimResult:
    scheduler: str
    results: list[RequestResult]
    makespan_ms: float
    steps: int
    useful_slot_steps: int
    total_slot_steps: int
    preemptions: int = 0
    max_batch_seen: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def throughput_tok_s(self) -> float:
        tokens = sum(r.output_tokens for r in self.results)
        return tokens / (self.makespan_ms / 1000) if self.makespan_ms else 0.0

    @property
    def slot_utilization(self) -> float:
        """Fraction of batch slots that produced a useful token (1 - padding waste)."""
        return self.useful_slot_steps / self.total_slot_steps if self.total_slot_steps else 0.0

    def _pct(self, values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    @property
    def mean_ttft_ms(self) -> float:
        return sum(r.ttft_ms for r in self.results) / len(self.results)

    @property
    def p99_latency_ms(self) -> float:
        return self._pct([r.latency_ms for r in self.results], 0.99)

    def summary(self) -> dict[str, float | str]:
        return {
            "scheduler": self.scheduler,
            "throughput_tok_s": round(self.throughput_tok_s, 1),
            "mean_ttft_ms": round(self.mean_ttft_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "slot_utilization": round(self.slot_utilization, 3),
            "max_batch": self.max_batch_seen,
            "preemptions": self.preemptions,
        }


def make_workload(
    n: int = 200,
    *,
    rate_per_s: float = 10.0,
    prompt_range: tuple[int, int] = (64, 1024),
    output_range: tuple[int, int] = (16, 512),
    seed: int = 0,
) -> list[Request]:
    """Poisson arrivals with uniformly mixed prompt and output lengths."""
    rng = random.Random(seed)
    t = 0.0
    requests = []
    for i in range(n):
        t += rng.expovariate(rate_per_s) * 1000
        requests.append(Request(i, t, rng.randint(*prompt_range), rng.randint(*output_range)))
    return requests


def simulate_static(
    requests: list[Request],
    *,
    kv_capacity_tokens: int,
    max_seq_len: int,
    max_batch_size: int = 256,
    cost: CostModel | None = None,
) -> SimResult:
    """Batch-at-a-time scheduling with contiguous, max-length KV reservations."""
    cost = cost or CostModel()
    batch_cap = min(max_batch_size, kv_capacity_tokens // max_seq_len)
    if batch_cap < 1:
        raise ValueError("KV cache cannot hold even one max-length sequence")
    waiting = deque(sorted(requests, key=lambda r: r.arrival_ms))
    t = 0.0
    results: list[RequestResult] = []
    steps = useful = total = max_batch = 0
    while waiting:
        t = max(t, waiting[0].arrival_ms)
        batch = []
        while waiting and waiting[0].arrival_ms <= t and len(batch) < batch_cap:
            batch.append(_Running(waiting.popleft()))
        max_batch = max(max_batch, len(batch))
        prefill = sum(r.request.prompt_tokens for r in batch)
        longest = max(r.request.output_tokens for r in batch)
        for step in range(longest):
            t += cost.step_ms(len(batch), prefill if step == 0 else 0)
            steps += 1
            total += len(batch)
            for r in batch:
                if r.generated < r.request.output_tokens:
                    r.generated += 1
                    useful += 1
                    if r.first_token_ms is None:
                        r.first_token_ms = t
                    if r.generated == r.request.output_tokens:
                        r.finish_ms = t  # streamed to the user, but its slot stays padded
        for r in batch:
            results.append(
                RequestResult(
                    r.request.id, r.request.arrival_ms, r.first_token_ms, r.finish_ms, r.generated
                )
            )
    return SimResult(
        "static",
        results,
        t,
        steps,
        useful,
        total,
        max_batch_seen=max_batch,
        notes=[f"batch capped at {batch_cap} by contiguous KV reservations"],
    )


def simulate_continuous(
    requests: list[Request],
    *,
    kv_capacity_tokens: int,
    block_size: int = 16,
    max_batch_size: int = 256,
    cost: CostModel | None = None,
) -> SimResult:
    """Iteration-level scheduling over a paged KV cache, with recompute preemption."""
    cost = cost or CostModel()
    too_long = [r.id for r in requests if r.prompt_tokens + r.output_tokens > kv_capacity_tokens]
    if too_long:
        raise ValueError(f"requests {too_long} can never fit in the KV cache")
    cache = PagedKVCache(kv_capacity_tokens // block_size, block_size)
    waiting = deque(sorted(requests, key=lambda r: r.arrival_ms))
    running: list[_Running] = []
    first_token: dict[int, float] = {}
    results: list[RequestResult] = []
    t = 0.0
    steps = useful = preemptions = max_batch = 0
    while waiting or running:
        # 1. Admit arrived requests while there is batch room and KV memory for the prompt.
        prefill = 0
        while (
            waiting
            and waiting[0].arrival_ms <= t
            and len(running) < max_batch_size
            and cache.can_allocate(waiting[0].prompt_tokens + 1)
        ):
            req = waiting.popleft()
            cache.add_sequence(req.id, req.prompt_tokens)
            running.append(_Running(req))
            prefill += req.prompt_tokens
        if not running:
            t = max(t, waiting[0].arrival_ms)
            continue

        # 2. Make sure every running sequence can store its next token; preempt if not.
        while sum(cache.needs_new_block(r.request.id) for r in running) > cache.allocator.num_free:
            victim = running.pop()  # newest first, like vLLM's recompute preemption
            cache.free(victim.request.id)
            waiting.appendleft(victim.request)
            prefill -= victim.request.prompt_tokens if victim.generated == 0 else 0
            preemptions += 1
        if not running:
            continue

        # 3. One forward step: every running sequence emits one token.
        max_batch = max(max_batch, len(running))
        t += cost.step_ms(len(running), max(prefill, 0))
        steps += 1
        still_running = []
        for r in running:
            cache.append_token(r.request.id)
            r.generated += 1
            useful += 1
            first_token.setdefault(r.request.id, t)
            if r.generated >= r.request.output_tokens:
                cache.free(r.request.id)
                results.append(
                    RequestResult(
                        r.request.id,
                        r.request.arrival_ms,
                        first_token[r.request.id],
                        t,
                        r.generated,
                    )
                )
            else:
                still_running.append(r)
        running = still_running
    results.sort(key=lambda r: r.id)
    return SimResult("continuous", results, t, steps, useful, useful, preemptions, max_batch)


def compare(
    requests: list[Request] | None = None,
    *,
    kv_capacity_tokens: int = 200_000,
    max_seq_len: int = 4096,
    cost: CostModel | None = None,
) -> list[SimResult]:
    """Run both schedulers on the same workload and KV budget."""
    requests = requests if requests is not None else make_workload()
    return [
        simulate_static(
            requests, kv_capacity_tokens=kv_capacity_tokens, max_seq_len=max_seq_len, cost=cost
        ),
        simulate_continuous(requests, kv_capacity_tokens=kv_capacity_tokens, cost=cost),
    ]
