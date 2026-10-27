"""Prefill/decode interference: chunked prefill and disaggregated serving.

Prefill is compute-bound and can take hundreds of milliseconds for a long prompt; decode
steps take ~15 ms. When both share a GPU, one 32k-token prompt arriving mid-stream stalls
*every* sequence that is decoding - users see their text freeze. Three ways to serve:

**Monolithic.** Prefill the whole prompt in one step. Best TTFT for that prompt, worst
inter-token latency (ITL) spike for everyone else.

**Chunked prefill** (vLLM default, SGLang ``--chunked-prefill-size``, TensorRT-LLM
chunked context). Give every step a token budget; a long prompt is prefilled a chunk at a
time, interleaved with decodes. The spike becomes a bounded slowdown; the long prompt's
TTFT grows slightly.

**Disaggregated prefill/decode** (SGLang PD mode, vLLM with KV connectors, NVIDIA
Dynamo). Prefill runs on separate GPUs and ships the KV cache to the decode GPUs. Decode
ITL is flat regardless of prompt traffic, and each pool can be sized and tuned on its
own; you pay for the extra pool, a KV transfer and a more complex deployment. (The
simulated decode throughput in this mode is for the decode GPU alone.)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .batching import CostModel


@dataclass(frozen=True)
class LongPrompt:
    arrival_ms: float
    tokens: int


@dataclass
class InterferenceResult:
    mode: str
    itl_p50_ms: float
    itl_p99_ms: float
    itl_max_ms: float
    mean_prompt_ttft_ms: float
    decode_tok_s: float

    def row(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "ITL p50 ms": round(self.itl_p50_ms, 1),
            "ITL p99 ms": round(self.itl_p99_ms, 1),
            "ITL max ms": round(self.itl_max_ms, 1),
            "long-prompt TTFT ms": round(self.mean_prompt_ttft_ms, 1),
            "decode tok/s": round(self.decode_tok_s, 0),
        }


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def default_prompts() -> list[LongPrompt]:
    """Six 32k-token prompts (long RAG contexts) arriving one per second."""
    return [LongPrompt(500 + 1000 * i, 32_000) for i in range(6)]


def simulate(
    mode: str = "monolithic",
    *,
    decoders: int = 32,
    decode_tokens: int = 500,
    prompts: list[LongPrompt] | None = None,
    chunk_tokens: int = 2048,
    kv_transfer_ms_per_1k_tokens: float = 2.0,
    cost: CostModel | None = None,
) -> InterferenceResult:
    """``decoders`` sequences stream tokens while long prompts arrive; measure who suffers.

    ``mode`` is ``monolithic``, ``chunked`` or ``disaggregated``.
    """
    if mode not in ("monolithic", "chunked", "disaggregated"):
        raise ValueError(f"unknown mode {mode!r}")
    cost = cost or CostModel()
    prompts = default_prompts() if prompts is None else prompts
    pending = deque(sorted(prompts, key=lambda p: p.arrival_ms))
    ttfts: list[float] = []

    if mode == "disaggregated":
        # A dedicated prefill GPU: prompts queue behind each other, never behind decodes.
        free_at = 0.0
        for p in pending:
            start = max(free_at, p.arrival_ms)
            free_at = start + cost.prefill_per_token_ms * p.tokens
            transfer = kv_transfer_ms_per_1k_tokens * p.tokens / 1000
            first_decode_step = cost.step_ms(decoders + 1)
            ttfts.append(free_at + transfer + first_decode_step - p.arrival_ms)
        pending = deque()

    t = 0.0
    itl: list[float] = []
    in_progress: list[list[float]] = []  # [arrival_ms, tokens_left]
    for _ in range(decode_tokens):
        while pending and pending[0].arrival_ms <= t:
            p = pending.popleft()
            in_progress.append([p.arrival_ms, p.tokens])
        budget = chunk_tokens if mode == "chunked" else float("inf")
        prefill_now = 0
        finished: list[float] = []
        for job in in_progress:
            take = int(min(job[1], budget - prefill_now))
            job[1] -= take
            prefill_now += take
            if job[1] == 0:
                finished.append(job[0])
            if prefill_now >= budget:
                break
        in_progress = [j for j in in_progress if j[1] > 0]
        dt = cost.step_ms(decoders, prefill_now)
        t += dt
        itl.append(dt)
        ttfts += [t - arrival for arrival in finished]

    return InterferenceResult(
        mode=mode,
        itl_p50_ms=_pct(itl, 0.50),
        itl_p99_ms=_pct(itl, 0.99),
        itl_max_ms=max(itl),
        mean_prompt_ttft_ms=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        decode_tok_s=decoders * decode_tokens / (t / 1000),
    )


def compare(**kwargs) -> list[InterferenceResult]:
    return [simulate(mode, **kwargs) for mode in ("monolithic", "chunked", "disaggregated")]
