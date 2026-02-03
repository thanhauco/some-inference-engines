"""An in-process fake of an OpenAI-compatible engine, so client code is testable without a GPU."""

from __future__ import annotations

import json

import httpx


def _words(payload: dict) -> list[str]:
    last = payload["messages"][-1]["content"]
    words = [w + " " for w in last.split()]
    return words[: payload.get("max_tokens", len(words))]


def _sse(payload: dict) -> bytes:
    model = payload["model"]
    words = _words(payload)
    lines = []
    for i, word in enumerate(words):
        finish = "stop" if i == len(words) - 1 else None
        chunk = {"model": model, "choices": [{"delta": {"content": word}, "finish_reason": finish}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    if payload.get("stream_options", {}).get("include_usage"):
        usage = {
            "prompt_tokens": 7,
            "completion_tokens": len(words),
            "total_tokens": 7 + len(words),
        }
        lines.append(f"data: {json.dumps({'model': model, 'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def handler(request: httpx.Request) -> httpx.Response:
    if request.headers.get("authorization") == "Bearer reject":
        return httpx.Response(401, json={"error": "bad key"})
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "fake-model"}]})
    if request.url.path.endswith("/chat/completions"):
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(
                200, content=_sse(payload), headers={"content-type": "text/event-stream"}
            )
        words = _words(payload)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "".join(words)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": len(words)},
            },
        )
    return httpx.Response(404)


transport = httpx.MockTransport(handler)
