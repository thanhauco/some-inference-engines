"""vLLM vs SGLang, side by side: how each one reuses prefixes and orders the queue.

Both engines avoid recomputing shared prompt prefixes, but they got there from opposite
directions, and the differences show up in agentic, multi-turn and few-shot workloads.

**vLLM - Automatic Prefix Caching (hash-based, block-granular).**
KV lives in fixed-size blocks (PagedAttention). A *full* block is identified by a hash of
its tokens chained with its parent block's hash, so the same hash means the same prefix.
A new request hashes its prompt block by block and reuses every leading block already in
the cache. Only full blocks are shared; a partial tail block is always recomputed. Freed
blocks stay cached and are evicted LRU when memory is needed.

**SGLang - RadixAttention (tree-based, token-granular).**
Cached KV is indexed by a radix tree of token ids (see :mod:`.radix_cache`). Matches are
exact to the token, and the tree makes "how much of *this* request is cached?" cheap to
ask for every waiting request - which enables...

**Cache-aware scheduling.** vLLM serves its waiting queue first-come-first-served (with
optional priorities). SGLang's default policy is *longest prefix match* (LPM): requests
whose prefix is already cached go first, so a cached prefix is used by all its waiting
requests before it can be evicted. Under memory pressure with many interleaved prefix
groups, FCFS thrashes the cache and LPM does not. (LPM trades strict fairness for hit
rate, and SGLang falls back to FCFS when the queue is very long, because matching every
waiting request gets expensive.)

The classes below model both caches with the same capacity so they can be compared
on the same request stream.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from .radix_cache import RadixCache


class BlockHashCache:
    """vLLM-style automatic prefix caching over full KV blocks, LRU-evicted."""

    def __init__(self, capacity_tokens: int, block_size: int = 16) -> None:
        self.block_size = block_size
        self.capacity_blocks = capacity_tokens // block_size
        self._blocks: OrderedDict[int, None] = OrderedDict()  # block hash -> (LRU order)
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def _block_hashes(self, tokens: Sequence[int]) -> list[int]:
        hashes, parent = [], 0
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            parent = hash((parent, tuple(tokens[start : start + self.block_size])))
            hashes.append(parent)
        return hashes

    def match_prefix(self, tokens: Sequence[int]) -> int:
        hit = 0
        for h in self._block_hashes(tokens):
            if h not in self._blocks:
                break
            hit += self.block_size
        return hit

    def serve(self, prompt: Sequence[int]) -> int:
        """Reuse leading cached blocks, compute the rest, cache the new full blocks."""
        hit, matching = 0, True
        for h in self._block_hashes(prompt):
            matching = matching and h in self._blocks
            if matching:
                hit += self.block_size
            self._blocks[h] = None
            self._blocks.move_to_end(h)
        while len(self._blocks) > self.capacity_blocks:
            self._blocks.popitem(last=False)
        self.prompt_tokens += len(prompt)
        self.cached_tokens += hit
        return len(prompt) - hit

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class RadixPrefixCache:
    """Adapter giving :class:`RadixCache` the same small interface as :class:`BlockHashCache`."""

    def __init__(self, capacity_tokens: int) -> None:
        self.cache = RadixCache(capacity_tokens)

    def match_prefix(self, tokens: Sequence[int]) -> int:
        return self.cache.match_prefix(tokens)

    def serve(self, prompt: Sequence[int]) -> int:
        return self.cache.serve(prompt)

    @property
    def hit_rate(self) -> float:
        return self.cache.counters.hit_rate


PrefixCache = BlockHashCache | RadixPrefixCache


def grouped_workload(
    groups: int = 8,
    per_group: int = 16,
    prefix_tokens: int = 1000,
    suffix_tokens: int = 60,
    seed: int = 0,
) -> list[list[int]]:
    """Requests from ``groups`` distinct long prefixes (e.g. different agents or few-shot
    templates), each with a unique suffix, arriving randomly interleaved."""
    rng = random.Random(seed)
    next_id = 0
    prompts = []
    for _ in range(groups):
        prefix = list(range(next_id, next_id + prefix_tokens))
        next_id += prefix_tokens
        for _ in range(per_group):
            prompts.append(prefix + list(range(next_id, next_id + suffix_tokens)))
            next_id += suffix_tokens
    rng.shuffle(prompts)
    return prompts


@dataclass
class ScheduleResult:
    cache: str
    policy: str
    hit_rate: float
    prefilled_tokens: int


def simulate_queue(
    prompts: Sequence[Sequence[int]],
    cache: PrefixCache,
    *,
    policy: str = "fcfs",
    batch_size: int = 8,
) -> ScheduleResult:
    """Drain a waiting queue in batches, choosing each batch by ``policy``.

    ``fcfs`` takes the oldest requests; ``lpm`` (SGLang's default) takes the requests
    with the longest cached prefix, oldest first among ties.
    """
    if policy not in ("fcfs", "lpm"):
        raise ValueError("policy must be 'fcfs' or 'lpm'")
    waiting = list(enumerate(prompts))
    prefilled = 0
    while waiting:
        if policy == "lpm":
            waiting.sort(key=lambda item: (-cache.match_prefix(item[1]), item[0]))
        batch, waiting = waiting[:batch_size], waiting[batch_size:]
        for _, prompt in batch:
            prefilled += cache.serve(prompt)
    name = "radix (SGLang)" if isinstance(cache, RadixPrefixCache) else "block-hash (vLLM)"
    return ScheduleResult(name, policy, round(cache.hit_rate, 3), prefilled)


def head_to_head(
    prompts: Sequence[Sequence[int]] | None = None,
    capacity_tokens: int = 3000,
    block_size: int = 16,
    batch_size: int = 8,
) -> list[ScheduleResult]:
    """vLLM-style (block hash + FCFS) vs SGLang-style (radix + LPM), plus the mixed cells."""
    prompts = prompts if prompts is not None else grouped_workload()
    results = []
    for policy in ("fcfs", "lpm"):
        results.append(
            simulate_queue(
                prompts,
                BlockHashCache(capacity_tokens, block_size),
                policy=policy,
                batch_size=batch_size,
            )
        )
        results.append(
            simulate_queue(
                prompts, RadixPrefixCache(capacity_tokens), policy=policy, batch_size=batch_size
            )
        )
    return results
