"""Capacity and cost: turning tokens/second into GPUs and dollars.

"Changing the inference engine can completely change the economics" - this module makes
that sentence computable. Feed it throughput you *measured* (``ie bench``), not numbers
from someone else's benchmark:

* cost per million output tokens = GPU $/hour / (tok/s x 3600 x utilization) x 1e6
* GPUs needed = peak tok/s / (tok/s per GPU x target utilization), rounded up
* migration break-even = one-off engineering cost / monthly GPU savings

Utilization matters as much as speed: a GPU that is 30% busy costs 3.3x more per token
than the same GPU at 100%, which is why continuous batching and good autoscaling often
beat a faster kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730


def cost_per_million_tokens(
    gpu_hourly_usd: float, tok_per_s: float, utilization: float = 1.0
) -> float:
    if tok_per_s <= 0 or not 0 < utilization <= 1:
        raise ValueError("tok_per_s must be positive and utilization in (0, 1]")
    return gpu_hourly_usd / (tok_per_s * 3600 * utilization) * 1e6


def gpus_needed(
    peak_tok_per_s: float, tok_per_s_per_gpu: float, target_utilization: float = 0.7
) -> int:
    """GPUs to serve the peak while keeping headroom (queues explode near 100% utilization)."""
    return max(1, math.ceil(peak_tok_per_s / (tok_per_s_per_gpu * target_utilization)))


@dataclass
class FleetPlan:
    name: str
    tok_per_s_per_gpu: float
    gpus: int
    monthly_usd: float
    usd_per_million_tokens: float

    def row(self) -> dict[str, float | str]:
        return {
            "option": self.name,
            "tok/s per GPU": round(self.tok_per_s_per_gpu),
            "GPUs": self.gpus,
            "$/month": round(self.monthly_usd),
            "$/1M tokens": round(self.usd_per_million_tokens, 3),
        }


def plan_fleet(
    name: str,
    tok_per_s_per_gpu: float,
    *,
    peak_tok_per_s: float,
    avg_tok_per_s: float | None = None,
    gpu_hourly_usd: float = 2.5,
    target_utilization: float = 0.7,
) -> FleetPlan:
    """Size a fleet for the peak and price it against the average load actually served."""
    avg = peak_tok_per_s * 0.5 if avg_tok_per_s is None else avg_tok_per_s
    gpus = gpus_needed(peak_tok_per_s, tok_per_s_per_gpu, target_utilization)
    monthly = gpus * gpu_hourly_usd * HOURS_PER_MONTH
    tokens_per_month = avg * 3600 * HOURS_PER_MONTH
    return FleetPlan(name, tok_per_s_per_gpu, gpus, monthly, monthly / tokens_per_month * 1e6)


def breakeven_months(migration_cost_usd: float, current: FleetPlan, candidate: FleetPlan) -> float:
    """Months until a migration pays for itself; ``inf`` if the candidate is not cheaper."""
    savings = current.monthly_usd - candidate.monthly_usd
    return migration_cost_usd / savings if savings > 0 else math.inf


def compare_options(
    measured_tok_per_s: dict[str, float],
    *,
    peak_tok_per_s: float,
    gpu_hourly_usd: float = 2.5,
    target_utilization: float = 0.7,
) -> list[FleetPlan]:
    """Fleet plans for several engines/configs given *your measured* per-GPU throughput."""
    return [
        plan_fleet(
            name,
            tps,
            peak_tok_per_s=peak_tok_per_s,
            gpu_hourly_usd=gpu_hourly_usd,
            target_utilization=target_utilization,
        )
        for name, tps in measured_tok_per_s.items()
    ]


EXAMPLE_MEASUREMENTS = {
    "engine A (baseline)": 2000.0,
    "engine B (+30% throughput)": 2600.0,
    "engine B + FP8 (+80%)": 3600.0,
}
"""Placeholder numbers for demos only - replace them with ``ie bench`` results."""
