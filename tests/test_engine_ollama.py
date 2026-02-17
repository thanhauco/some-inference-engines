import json

import httpx
import pytest

from inference_engines.config import get_profile
from inference_engines.engines import ollama

PULL_EVENTS = [
    {"status": "pulling manifest"},
    {"status": "downloading", "completed": 5, "total": 10},
    {"status": "success"},
]


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    assert not path.startswith("/v1"), "native API lives at the server root"
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    if path == "/api/pull":
        body = "\n".join(json.dumps(e) for e in PULL_EVENTS)
        return httpx.Response(200, content=body.encode())
    if path == "/api/generate":
        return httpx.Response(
            200, json={"response": "hi", "eval_count": 50, "eval_duration": 2_000_000_000}
        )
    if path == "/api/ps":
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b", "size_vram": 5e9}]})
    return httpx.Response(404)


@pytest.fixture
def native():
    client = ollama.OllamaNative(
        get_profile("ollama", env={}), transport=httpx.MockTransport(handler)
    )
    yield client
    client.close()


def test_ensure_model_skips_present_models(native):
    assert native.ensure_model("llama3.1:8b") is False


def test_ensure_model_pulls_missing_models(native):
    assert list(native.pull("qwen2.5:7b"))[-1] == {"status": "success"}
    assert native.ensure_model("qwen2.5:7b") is True


def test_generate_and_tokens_per_second(native):
    response = native.generate("llama3.1:8b", "hello", temperature=0)
    assert ollama.tokens_per_second(response) == pytest.approx(25.0)
    assert native.running_models()[0]["name"] == "llama3.1:8b"


def test_modelfile_rendering():
    text = ollama.modelfile("llama3.1:8b", system="Be brief.", temperature=0.2, num_ctx=8192)
    assert text.splitlines() == [
        "FROM llama3.1:8b",
        "PARAMETER temperature 0.2",
        "PARAMETER num_ctx 8192",
        'SYSTEM """Be brief."""',
    ]
    assert ollama.pull_command("llama3.1:8b") == ["ollama", "pull", "llama3.1:8b"]
