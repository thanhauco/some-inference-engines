import asyncio

import httpx
import pytest
from fake_engine import handler, transport

from inference_engines.client import (
    AsyncInferenceClient,
    InferenceClient,
    InferenceError,
    parse_sse_lines,
)
from inference_engines.config import get_profile

MESSAGES = [{"role": "user", "content": "the model may stay the same"}]


def make_client(**kw):
    return InferenceClient(get_profile("vllm", env={}), transport=transport, **kw)


def test_parse_sse_stops_at_done():
    lines = [
        'data: {"a": 1}',
        "",
        ": keep-alive",
        'data: {"a": 2}',
        "data: [DONE]",
        'data: {"a": 3}',
    ]
    assert [c["a"] for c in parse_sse_lines(lines)] == [1, 2]


def test_models_and_health():
    with make_client() as client:
        assert client.models() == ["fake-model"]
        assert client.health()


def test_chat_non_streaming():
    with make_client() as client:
        result = client.chat(MESSAGES)
    assert result.text.strip() == "the model may stay the same"
    assert result.completion_tokens == 6
    assert result.finish_reason == "stop"


def test_stream_chat_yields_deltas_and_timings():
    with make_client() as client:
        deltas = list(client.stream_chat(MESSAGES))
        completion = client.timed_stream(MESSAGES, max_tokens=3)
    assert "".join(deltas).strip() == "the model may stay the same"
    assert completion.completion_tokens == 3
    assert completion.ttft_s is not None and completion.ttft_s <= completion.latency_s
    assert len(completion.chunk_times_s) == 3
    assert completion.tpot_s is not None


def test_errors_are_raised_with_status():
    profile = get_profile("ollama", env={"IE_OLLAMA_API_KEY": "reject"})
    with InferenceClient(profile, transport=transport) as client:
        with pytest.raises(InferenceError, match="401"):
            client.chat(MESSAGES)
        assert not client.health()


def test_async_client_streams():
    async def run():
        async with AsyncInferenceClient(
            get_profile("sglang", env={}), transport=httpx.MockTransport(handler)
        ) as client:
            completion = await client.timed_stream(MESSAGES)
            deltas = [d async for d in client.stream_deltas(MESSAGES, max_tokens=2)]
        return completion, deltas

    completion, deltas = asyncio.run(run())
    assert completion.text.strip() == "the model may stay the same"
    assert deltas == ["the ", "model "]
