"""Latency and throughput statistics for a benchmark run.

* **TTFT** (time to first token): queueing + prefill. What a chat user feels as "lag".
* **TPOT** (time per output token, after the first): decode speed. What streaming feels like.
* **E2E latency**: TTFT + (tokens - 1) x TPOT.
* **Throughput**: output tokens per second across all requests - what you pay for.

Report percentiles, not means: p99 TTFT is what your unluckiest 1% of users see, and it
moves first when a server saturates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class RequestRecord:
    start_s: float
    latency_s: float
    ttft_s: float | None
    output_tokens: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass
class Summary:
    requests: int
    errors: int
    duration_s: float
    request_throughput: float
    output_token_throughput: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    e2e_ms: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "duration_s": round(self.duration_s, 3),
            "req_per_s": round(self.request_throughput, 3),
            "output_tok_per_s": round(self.output_token_throughput, 1),
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
        }


def _dist_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(1000 * sum(values) / len(values), 2) if values else math.nan,
        "p50": round(1000 * percentile(values, 50), 2),
        "p90": round(1000 * percentile(values, 90), 2),
        "p99": round(1000 * percentile(values, 99), 2),
    }


def summarize(records: Sequence[RequestRecord], duration_s: float | None = None) -> Summary:
    ok = [r for r in records if r.ok]
    if duration_s is None:
        duration_s = (
            max(r.start_s + r.latency_s for r in records) - min(r.start_s for r in records)
            if records
            else 0.0
        )
    tokens = sum(r.output_tokens for r in ok)
    return Summary(
        requests=len(records),
        errors=len(records) - len(ok),
        duration_s=duration_s,
        request_throughput=len(ok) / duration_s if duration_s else 0.0,
        output_token_throughput=tokens / duration_s if duration_s else 0.0,
        ttft_ms=_dist_ms([r.ttft_s for r in ok if r.ttft_s is not None]),
        tpot_ms=_dist_ms([t for r in ok if (t := r.tpot_s) is not None]),
        e2e_ms=_dist_ms([r.latency_s for r in ok]),
    )


def goodput(
    records: Sequence[RequestRecord], duration_s: float, *, ttft_slo_s: float, tpot_slo_s: float
) -> float:
    """Requests per second that met *both* latency SLOs - throughput that users actually feel."""
    good = [
        r for r in records
        if r.ok and r.ttft_s is not None and r.ttft_s <= ttft_slo_s
        and (r.tpot_s is None or r.tpot_s <= tpot_slo_s)
    ]  # fmt: skip
    return len(good) / duration_s if duration_s else 0.0
