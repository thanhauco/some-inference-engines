"""Speculative decoding: a cheap draft model guesses, the big target model verifies.

Decoding one token at a time leaves the GPU mostly idle (it is waiting on memory). If a
small *draft* model proposes ``k`` tokens, the *target* model can score all ``k`` in a
single forward pass - about the cost of generating one token. Each proposed token is
accepted with probability ``min(1, p(x) / q(x))`` (``p`` = target, ``q`` = draft); at the
first rejection a corrected token is sampled from ``max(0, p - q)``. If all ``k`` pass,
a bonus token comes free from the target's last distribution.

The key property (Leviathan et al., 2023; Chen et al., 2023): the output distribution is
*exactly* the target model's. You trade extra draft compute for fewer target passes,
and the speedup depends on how often the draft agrees with the target (acceptance rate).

SGLang ships this as EAGLE speculative decoding (``--speculative-algorithm EAGLE``),
where the draft is a light head on the target's own hidden states; the algorithm below is
the general draft-model version with tiny numpy "models".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max())
    return z / z.sum()


class ToyLM:
    """Deterministic stand-in for a language model: next-token probs depend on the last 2 tokens."""

    def __init__(self, vocab_size: int = 32, seed: int = 0, sharpness: float = 2.5) -> None:
        self.vocab_size = vocab_size
        self.seed = seed
        self.sharpness = sharpness
        self.calls = 0

    def probs(self, context: Sequence[int]) -> np.ndarray:
        a = context[-1] if context else 0
        b = context[-2] if len(context) > 1 else 0
        rng = np.random.default_rng((self.seed * 1_000_003 + a * 1_009 + b * 7) % 2**32)
        return _softmax(rng.normal(size=self.vocab_size) * self.sharpness)

    def forward(self, context: Sequence[int], positions: int = 1) -> list[np.ndarray]:
        """One (batched) forward pass returning next-token distributions for the last positions.

        ``positions=k+1`` scores the k drafted tokens at once - that is the whole trick.
        """
        self.calls += 1
        n = len(context)
        return [self.probs(context[: n - positions + 1 + i]) for i in range(positions)]


class DraftLM(ToyLM):
    """A draft model that agrees with ``target`` to a tunable degree (``agreement`` in [0, 1])."""

    def __init__(self, target: ToyLM, agreement: float = 0.8, seed: int = 99) -> None:
        super().__init__(target.vocab_size, seed, target.sharpness)
        self.target = target
        self.agreement = agreement

    def probs(self, context: Sequence[int]) -> np.ndarray:
        own = super().probs(context)
        return self.agreement * self.target.probs(context) + (1 - self.agreement) * own


@dataclass
class SpecStats:
    tokens: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self) -> float:
        return self.tokens / self.target_calls if self.target_calls else 0.0


def speculative_step(
    context: list[int],
    target: ToyLM,
    draft: ToyLM,
    k: int,
    rng: np.random.Generator,
    stats: SpecStats | None = None,
) -> list[int]:
    """Propose ``k`` draft tokens, verify them in one target pass, return the accepted tokens."""
    stats = stats if stats is not None else SpecStats()
    drafted: list[int] = []
    draft_dists: list[np.ndarray] = []
    for _ in range(k):
        q = draft.forward(context + drafted)[0]
        stats.draft_calls += 1
        token = int(rng.choice(len(q), p=q))
        drafted.append(token)
        draft_dists.append(q)

    target_dists = target.forward(context + drafted, positions=k + 1)
    stats.target_calls += 1
    stats.proposed += k

    out: list[int] = []
    for token, q, p in zip(drafted, draft_dists, target_dists[:k], strict=True):
        if rng.random() < min(1.0, p[token] / q[token]):
            out.append(token)
            stats.accepted += 1
            continue
        residual = np.maximum(p - q, 0)
        residual = residual / residual.sum() if residual.sum() > 0 else p
        out.append(int(rng.choice(len(p), p=residual)))
        break
    else:
        bonus = target_dists[k]
        out.append(int(rng.choice(len(bonus), p=bonus)))
    stats.tokens += len(out)
    return out


def generate(
    prompt: list[int],
    max_new_tokens: int,
    target: ToyLM,
    draft: ToyLM | None = None,
    k: int = 4,
    seed: int = 0,
) -> tuple[list[int], SpecStats]:
    """Generate with speculative decoding, or plain autoregressive decoding if ``draft`` is None."""
    rng = np.random.default_rng(seed)
    stats = SpecStats()
    out = list(prompt)
    while len(out) - len(prompt) < max_new_tokens:
        if draft is None:
            p = target.forward(out)[0]
            stats.target_calls += 1
            stats.tokens += 1
            out.append(int(rng.choice(len(p), p=p)))
        else:
            out.extend(speculative_step(out, target, draft, k, rng, stats))
    return out[: len(prompt) + max_new_tokens], stats


def expected_tokens_per_step(alpha: float, k: int) -> float:
    """Expected tokens per target pass for acceptance rate ``alpha``: (1 - a^(k+1)) / (1 - a)."""
    if alpha >= 1:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, draft_cost: float) -> float:
    """Wall-clock speedup vs plain decoding; ``draft_cost`` = draft pass time / target pass time."""
    return expected_tokens_per_step(alpha, k) / (k * draft_cost + 1)


def best_k(alpha: float, draft_cost: float, max_k: int = 16) -> int:
    return max(range(1, max_k + 1), key=lambda k: expected_speedup(alpha, k, draft_cost))
