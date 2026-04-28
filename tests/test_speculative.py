import numpy as np
import pytest

from inference_engines.concepts.speculative import (
    DraftLM,
    SpecStats,
    ToyLM,
    best_k,
    expected_speedup,
    expected_tokens_per_step,
    generate,
    speculative_step,
)


def test_perfect_draft_accepts_everything():
    target = ToyLM(seed=1)
    tokens, stats = generate([1, 2], 100, target, DraftLM(target, agreement=1.0), k=4)
    assert len(tokens) == 102
    assert stats.acceptance_rate == 1.0
    assert stats.tokens_per_target_call == pytest.approx(5.0)


def test_worse_draft_means_fewer_tokens_per_target_call():
    target = ToyLM(seed=1)
    rates = []
    for agreement in (0.95, 0.6, 0.1):
        _, stats = generate([1, 2], 300, target, DraftLM(target, agreement), k=4)
        rates.append(stats.tokens_per_target_call)
    assert rates[0] > rates[1] > rates[2] >= 1.0


def test_plain_decoding_uses_one_target_call_per_token():
    _, stats = generate([0], 50, ToyLM())
    assert stats.target_calls == 50


def test_output_distribution_matches_target_exactly():
    """The defining property: the first emitted token is distributed as p, whatever q is."""
    target = ToyLM(vocab_size=8, seed=3, sharpness=1.0)
    draft = DraftLM(target, agreement=0.3)
    context = [4, 5]
    rng = np.random.default_rng(0)
    n = 20_000
    counts = np.zeros(8)
    for _ in range(n):
        counts[speculative_step(context, target, draft, 3, rng, SpecStats())[0]] += 1
    np.testing.assert_allclose(counts / n, target.probs(context), atol=0.015)


def test_expected_speedup_formula():
    assert expected_tokens_per_step(1.0, 4) == 5
    assert expected_tokens_per_step(0.0, 4) == 1
    assert expected_tokens_per_step(0.8, 4) == pytest.approx((1 - 0.8**5) / 0.2)
    assert expected_speedup(0.8, 4, draft_cost=0.05) > 2.5
    assert expected_speedup(0.2, 8, draft_cost=0.3) < 1  # a bad draft slows you down
    assert best_k(0.9, 0.02) > best_k(0.5, 0.02)
