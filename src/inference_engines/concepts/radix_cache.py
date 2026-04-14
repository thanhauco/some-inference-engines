"""RadixAttention, SGLang's automatic KV-cache reuse across requests.

Agentic and multi-call programs send the same prefix again and again: the system prompt,
tool definitions, few-shot examples, the conversation so far, the shared trunk of a
``fork``. Recomputing that prefix (prefill) for every call wastes GPU time and inflates
time-to-first-token.

SGLang keeps the KV cache of finished requests in a **radix tree** keyed by token ids.
A new request walks the tree, reuses the KV of the longest matching prefix and only
prefills the rest. When memory runs low, least-recently-used leaves are evicted first,
so hot prefixes (the system prompt) stay resident while one-off tails go.

This is a compressed radix tree (edges carry token runs) with LRU leaf eviction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(eq=False)
class _Node:
    key: tuple[int, ...] = ()
    parent: _Node | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    last_access: int = 0


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class CacheCounters:
    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    evicted_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class RadixCache:
    def __init__(self, capacity_tokens: int | None = None) -> None:
        self.capacity_tokens = capacity_tokens
        self.root = _Node()
        self.total_tokens = 0
        self.counters = CacheCounters()
        self._clock = 0

    def _touch(self, node: _Node) -> None:
        self._clock += 1
        while node is not None:
            node.last_access = self._clock
            node = node.parent

    # -- lookups ------------------------------------------------------------------------
    def match_prefix(self, tokens: Sequence[int]) -> int:
        """Length of the longest cached prefix of ``tokens``."""
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            k = _common_prefix(child.key, tokens[i:])
            i += k
            self._touch(child)
            if k < len(child.key):
                break
            node = child
        return i

    def insert(self, tokens: Sequence[int]) -> int:
        """Cache the KV of ``tokens``; returns how many tokens were newly stored."""
        tokens = tuple(tokens)
        node, i, added = self.root, 0, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                leaf = _Node(key=tokens[i:], parent=node)
                node.children[tokens[i]] = leaf
                added = len(leaf.key)
                self._touch(leaf)
                break
            k = _common_prefix(child.key, tokens[i:])
            if k < len(child.key):
                # Split the edge: node -> mid (shared part) -> child (old remainder).
                mid = _Node(key=child.key[:k], parent=node)
                node.children[tokens[i]] = mid
                child.key = child.key[k:]
                child.parent = mid
                mid.children[child.key[0]] = child
                child = mid
            self._touch(child)
            node, i = child, i + k
        self.total_tokens += added
        if self.capacity_tokens is not None and self.total_tokens > self.capacity_tokens:
            self.evict(self.total_tokens - self.capacity_tokens)
        return added

    def serve(self, prompt: Sequence[int]) -> int:
        """Handle one request: reuse the cached prefix, prefill the rest, cache the prompt.

        Returns the number of tokens that actually had to be prefilled.
        """
        hit = self.match_prefix(prompt)
        self.counters.requests += 1
        self.counters.prompt_tokens += len(prompt)
        self.counters.cached_tokens += hit
        self.insert(prompt)
        return len(prompt) - hit

    # -- eviction -----------------------------------------------------------------------
    def _leaves(self) -> list[_Node]:
        leaves, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node is not self.root:
                leaves.append(node)
        return leaves

    def evict(self, num_tokens: int) -> int:
        """Drop least-recently-used leaves until ``num_tokens`` tokens are freed."""
        freed = 0
        while freed < num_tokens:
            leaves = self._leaves()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            del victim.parent.children[victim.key[0]]
            freed += len(victim.key)
        self.total_tokens -= freed
        self.counters.evicted_tokens += freed
        return freed

    def num_nodes(self) -> int:
        count, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count - 1


def agent_workload(
    num_sessions: int = 8,
    turns: int = 5,
    system_prompt_tokens: int = 1500,
    turn_tokens: int = 120,
) -> list[list[int]]:
    """Prompts of a typical agent: shared system prompt + tools, then growing per-session history.

    Token ids are synthetic but structured like the real thing: every prompt starts with
    the same system prompt, and each turn of a session extends that session's previous prompt.
    """
    system = list(range(system_prompt_tokens))
    prompts = []
    next_id = system_prompt_tokens
    histories = [list(system) for _ in range(num_sessions)]
    for _ in range(turns):
        for history in histories:
            history.extend(range(next_id, next_id + turn_tokens))
            next_id += turn_tokens
            prompts.append(list(history))
    return prompts


def prefill_savings(
    prompts: list[list[int]], capacity_tokens: int | None = None
) -> dict[str, float]:
    cache = RadixCache(capacity_tokens)
    prefilled = sum(cache.serve(p) for p in prompts)
    total = sum(len(p) for p in prompts)
    return {
        "prompt_tokens": total,
        "prefilled_tokens": prefilled,
        "hit_rate": round(cache.counters.hit_rate, 3),
        "prefill_reduction_x": round(total / prefilled, 2) if prefilled else float("inf"),
    }
