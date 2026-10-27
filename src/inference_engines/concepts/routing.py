"""Cache-aware routing: prefix reuse at cluster scale.

One GPU reuses a prefix only if the request lands on *that* GPU. Behind a load balancer
with N replicas, round-robin sends consecutive requests of the same agent to different
replicas, so each replica must compute (and hold) every prefix: the hit rate collapses
and the caches thrash.

Routers built for LLMs fix this by tracking which prefixes each replica holds:

* **Prefix affinity** (SGLang's ``sgl-router`` cache-aware policy, vLLM production-stack,
  llm-d / Gateway API inference extension): send the request to the replica with the
  longest cached prefix - unless that replica is overloaded, then fall back to the least
  loaded one. The balance guard matters: pure affinity can pile a hot prefix on one GPU.
* **Consistent hashing** on the first tokens (or a session id): stateless and simple;
  works when the prefix is identified by its head, but cannot rebalance a hot key.

The simulation uses one radix cache per replica, the same structure the engine uses.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from .prefix_caching import grouped_workload
from .radix_cache import RadixCache

POLICIES = ("round_robin", "consistent_hash", "prefix_affinity")


@dataclass
class RoutingResult:
    policy: str
    hit_rate: float
    prefilled_tokens: int
    load_imbalance: float
    """Requests on the busiest replica / mean requests per replica (1.0 = perfectly even)."""

    def row(self) -> dict[str, float | str]:
        return {
            "policy": self.policy,
            "hit_rate": round(self.hit_rate, 3),
            "prefilled_tokens": self.prefilled_tokens,
            "load_imbalance": round(self.load_imbalance, 2),
        }


def _stable_hash(tokens: Sequence[int]) -> int:
    digest = hashlib.blake2b(repr(tuple(tokens)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simulate_routing(
    prompts: Sequence[Sequence[int]],
    *,
    policy: str = "prefix_affinity",
    replicas: int = 4,
    capacity_tokens: int = 4000,
    hash_prefix_tokens: int = 64,
    balance_threshold: float = 1.5,
) -> RoutingResult:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    caches = [RadixCache(capacity_tokens) for _ in range(replicas)]
    work = [0] * replicas  # prefilled tokens per replica
    served = [0] * replicas  # requests per replica
    for i, prompt in enumerate(prompts):
        if policy == "round_robin":
            target = i % replicas
        elif policy == "consistent_hash":
            target = _stable_hash(prompt[:hash_prefix_tokens]) % replicas
        else:
            matches = [cache.match_prefix(prompt) for cache in caches]
            best = max(range(replicas), key=lambda r: (matches[r], -served[r]))
            least = min(range(replicas), key=lambda r: served[r])
            # Balance guard: don't let affinity pile a hot prefix onto one replica.
            overloaded = served[best] > balance_threshold * (served[least] + 1)
            target = least if matches[best] == 0 or overloaded else best
        work[target] += caches[target].serve(prompt)
        served[target] += 1
    total = sum(len(p) for p in prompts)
    prefilled = sum(work)
    mean_served = sum(served) / replicas
    return RoutingResult(
        policy=policy,
        hit_rate=1 - prefilled / total if total else 0.0,
        prefilled_tokens=prefilled,
        load_imbalance=max(served) / mean_served if mean_served else 1.0,
    )


def compare_policies(
    prompts: Sequence[Sequence[int]] | None = None, **kwargs
) -> list[RoutingResult]:
    """All routing policies on the same request stream (16 agents x 24 requests by default)."""
    if prompts is None:
        prompts = grouped_workload(groups=16, per_group=24, prefix_tokens=1000, suffix_tokens=60)
    return [simulate_routing(prompts, policy=policy, **kwargs) for policy in POLICIES]
