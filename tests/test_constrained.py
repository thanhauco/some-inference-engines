import json

import pytest

from inference_engines.concepts.constrained import (
    EnumField,
    Literal,
    StringField,
    compile_schema,
    generate_json,
    random_char_model,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string", "maxLength": 10},
        "priority": {"type": "integer"},
        "category": {"enum": ["billing", "bug", "feature_request"]},
        "escalate": {"type": "boolean"},
    },
}


def test_compile_schema_merges_literals():
    segments = compile_schema(SCHEMA)
    assert segments[0] == Literal('{"customer": "')
    assert isinstance(segments[1], StringField)
    assert segments[2] == Literal('", "priority": ')
    assert segments[-1] == Literal("}")
    assert EnumField(('"billing"', '"bug"', '"feature_request"')) in segments


@pytest.mark.parametrize("seed", range(25))
def test_even_a_random_model_produces_schema_valid_json(seed):
    text, _ = generate_json(SCHEMA, random_char_model(seed), seed=seed)
    obj = json.loads(text)
    assert list(obj) == ["customer", "priority", "category", "escalate"]
    assert isinstance(obj["customer"], str) and 1 <= len(obj["customer"]) <= 10
    assert isinstance(obj["priority"], int)
    assert obj["category"] in {"billing", "bug", "feature_request"}
    assert isinstance(obj["escalate"], bool)


def test_jump_forward_skips_most_model_calls():
    model = random_char_model(7)
    fast, fast_stats = generate_json(SCHEMA, model, jump_forward=True, seed=1)
    slow, slow_stats = generate_json(SCHEMA, model, jump_forward=False, seed=1)
    json.loads(fast), json.loads(slow)
    assert fast_stats.forced_chars > 40
    assert slow_stats.forced_chars == 0
    assert slow_stats.model_calls > 2 * fast_stats.model_calls


def test_unsupported_schema():
    with pytest.raises(ValueError):
        compile_schema({"type": "array"})
    with pytest.raises(ValueError):
        compile_schema({"type": "object", "properties": {"x": {"type": "array"}}})
