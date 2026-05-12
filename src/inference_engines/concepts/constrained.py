"""Constrained decoding with jump-forward, how SGLang guarantees structured output fast.

To force valid JSON, the engine compiles the schema into a finite-state machine and, at
every step, **masks** the model's logits so only tokens the FSM allows can be sampled.
The output is valid by construction - no retries, no "please answer in JSON".

SGLang adds **jump-forward decoding** on a *compressed* FSM: whenever the FSM has a single
legal continuation (``{"name": "`` is fixed by the schema; so is the rest of ``"billing"``
once ``"bi`` is typed and no other enum value starts that way), those characters are
appended directly, with no model forward pass at all.

This module implements both over a character-level toy model for a JSON-schema subset
(object of string / integer / boolean / enum properties, all required, in order).
"""

from __future__ import annotations

import json
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

END = "\0"
"""Pseudo-character a model emits to end an unquoted field (integers)."""

VOCAB = tuple(string.ascii_letters + string.digits + " -_.,:;!?'\"{}") + (END,)

CharModel = Callable[[str], np.ndarray]
"""A 'model': given the text so far, return one logit per character of :data:`VOCAB`."""


@dataclass(frozen=True)
class Literal:
    text: str


@dataclass(frozen=True)
class StringField:
    max_len: int = 24


@dataclass(frozen=True)
class IntegerField:
    max_digits: int = 6
    """Includes the sign; must be at least 2 so ``-`` can always be followed by a digit."""


@dataclass(frozen=True)
class EnumField:
    choices: tuple[str, ...]
    """Prefix-free alternatives (JSON-encoded strings always are, thanks to the quotes)."""


Segment = Literal | StringField | IntegerField | EnumField


def compile_schema(schema: dict[str, Any]) -> list[Segment]:
    """Compile a flat JSON schema into fixed literals and free fields (a linear FSM)."""
    if schema.get("type") != "object":
        raise ValueError("only object schemas are supported")
    segments: list[Segment] = []
    props = schema.get("properties", {})
    for i, (name, spec) in enumerate(props.items()):
        prefix = "{" if i == 0 else ", "
        segments.append(Literal(f'{prefix}"{name}": '))
        if "enum" in spec:
            segments.append(EnumField(tuple(json.dumps(v) for v in spec["enum"])))
        elif spec.get("type") == "string":
            segments.append(Literal('"'))
            segments.append(StringField(spec.get("maxLength", 24)))
            segments.append(Literal('"'))
        elif spec.get("type") == "integer":
            segments.append(IntegerField())
        elif spec.get("type") == "boolean":
            segments.append(EnumField(("true", "false")))
        else:
            raise ValueError(f"unsupported property type for {name!r}: {spec}")
    segments.append(Literal("}" if props else "{}"))
    return _merge_literals(segments)


def _merge_literals(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if isinstance(seg, Literal) and merged and isinstance(merged[-1], Literal):
            merged[-1] = Literal(merged[-1].text + seg.text)
        else:
            merged.append(seg)
    return merged


@dataclass
class DecodeStats:
    model_calls: int = 0
    forced_chars: int = 0
    """Characters appended by jump-forward without calling the model."""


_STRING_CHARS = frozenset(c for c in VOCAB if c not in {'"', END, "\\"})
_DIGITS = frozenset(string.digits)


def _sample(
    model: CharModel, text: str, allowed: set[str], rng: np.random.Generator, stats: DecodeStats
) -> str:
    """One model call with every disallowed character masked to -inf."""
    stats.model_calls += 1
    logits = np.asarray(model(text), dtype=float)
    mask = np.array([c in allowed for c in VOCAB])
    masked = np.where(mask, logits, -np.inf)
    probs = np.exp(masked - masked[mask].max())
    probs /= probs.sum()
    return VOCAB[int(rng.choice(len(VOCAB), p=probs))]


def generate_json(
    schema: dict[str, Any],
    model: CharModel,
    *,
    jump_forward: bool = True,
    seed: int = 0,
) -> tuple[str, DecodeStats]:
    """Decode a JSON object that is guaranteed to match ``schema``."""
    rng = np.random.default_rng(seed)
    stats = DecodeStats()
    text = ""

    def emit_forced(chars: str) -> None:
        nonlocal text
        if jump_forward:
            stats.forced_chars += len(chars)
            text += chars
        else:  # without jump-forward every forced char still costs a (fully masked) model call
            for c in chars:
                text += _sample(model, text, {c}, rng, stats)

    for seg in compile_schema(schema):
        if isinstance(seg, Literal):
            emit_forced(seg.text)
        elif isinstance(seg, StringField):
            value = ""
            while len(value) < seg.max_len:
                allowed = set(_STRING_CHARS) | ({'"'} if value else set())
                c = _sample(model, text, allowed, rng, stats)
                if c == '"':
                    break  # the closing quote is the next literal; don't emit it twice
                value += c
                text += c
        elif isinstance(seg, IntegerField):
            digits = ""
            while len(digits) < seg.max_digits:
                if digits == "0":
                    break  # a lone zero is complete: no leading zeros
                if digits == "":
                    allowed = set(_DIGITS) | {"-"}
                elif digits == "-":
                    allowed = set(_DIGITS) - {"0"}
                else:
                    allowed = set(_DIGITS) | {END}
                c = _sample(model, text, allowed, rng, stats)
                if c == END:
                    break
                digits += c
                text += c
        elif isinstance(seg, EnumField):
            typed = ""
            while True:
                remaining = [c for c in seg.choices if c.startswith(typed)]
                if typed in seg.choices and len(remaining) == 1:
                    break
                rest = _common_tail(remaining, len(typed))
                if rest:  # every remaining choice continues the same way: jump forward
                    emit_forced(rest)
                    typed += rest
                    continue
                allowed = {c[len(typed)] for c in remaining if len(c) > len(typed)}
                c = _sample(model, text, allowed, rng, stats)
                typed += c
                text += c
    return text, stats


def _common_tail(choices: list[str], start: int) -> str:
    """Characters after ``start`` shared by all choices (the deterministic part of the trie)."""
    if not choices:
        return ""
    tails = [c[start:] for c in choices]
    n = min(len(t) for t in tails)
    out = ""
    for i in range(n):
        ch = tails[0][i]
        if all(t[i] == ch for t in tails):
            out += ch
        else:
            break
    return out


def random_char_model(seed: int = 0) -> CharModel:
    """An untrained 'model' - random logits. Even it produces valid JSON under constraints."""
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(64, len(VOCAB))) * 2

    def model(text: str) -> np.ndarray:
        return table[sum((i + 1) * ord(c) for i, c in enumerate(text[-3:])) % 64]

    return model
