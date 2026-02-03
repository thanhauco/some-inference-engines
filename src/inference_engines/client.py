"""One client for four engines.

Ollama, vLLM, SGLang and TensorRT-LLM (``trtllm-serve``) all implement the OpenAI
``/v1/chat/completions`` API, including server-sent-event streaming. This module talks
that API directly with ``httpx`` so application code never imports an engine.

Streaming responses are also where latency lives, so every call records:

* **TTFT** - time to first token (queueing + prefill)
* **latency** - time until the last token (TTFT + decode)
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import EngineProfile, get_profile

Message = dict[str, str]


@dataclass
class Completion:
    text: str
    model: str
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    chunk_times_s: list[float] = field(default_factory=list)
    """Arrival time (relative to request start) of every streamed content chunk."""

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first one (the decode speed)."""
        tokens = self.completion_tokens or len(self.chunk_times_s)
        if self.ttft_s is None or tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (tokens - 1)


class InferenceError(RuntimeError):
    pass


def parse_sse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield the JSON payloads of an OpenAI-style SSE stream, stopping at ``[DONE]``."""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


def build_chat_payload(
    model: str,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        # Ask for a final usage chunk so token counts are exact rather than estimated.
        payload["stream_options"] = {"include_usage": True}
    payload.update(extra)
    return payload


class _StreamAccumulator:
    """Folds streamed chunks into a :class:`Completion` while timing them."""

    def __init__(self, model: str, start: float) -> None:
        self.model = model
        self.start = start
        self.parts: list[str] = []
        self.chunk_times: list[float] = []
        self.usage: dict[str, Any] = {}
        self.finish_reason: str | None = None

    def add(self, chunk: dict[str, Any]) -> str:
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        delta_text = ""
        for choice in chunk.get("choices") or []:
            delta_text += (choice.get("delta") or {}).get("content") or ""
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
        if delta_text:
            self.parts.append(delta_text)
            self.chunk_times.append(time.perf_counter() - self.start)
        return delta_text

    def result(self) -> Completion:
        return Completion(
            text="".join(self.parts),
            model=self.model,
            latency_s=time.perf_counter() - self.start,
            ttft_s=self.chunk_times[0] if self.chunk_times else None,
            prompt_tokens=self.usage.get("prompt_tokens"),
            completion_tokens=self.usage.get("completion_tokens"),
            finish_reason=self.finish_reason,
            chunk_times_s=self.chunk_times,
        )


def _completion_from_json(body: dict[str, Any], model: str, latency: float) -> Completion:
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    return Completion(
        text=(choice.get("message") or {}).get("content") or "",
        model=body.get("model") or model,
        latency_s=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise InferenceError(f"{response.status_code} from {response.request.url}: {response.text}")


def _headers(profile: EngineProfile) -> dict[str, str]:
    return {"Authorization": f"Bearer {profile.api_key}"}


class InferenceClient:
    """Synchronous OpenAI-compatible client bound to one engine profile."""

    def __init__(
        self,
        profile: EngineProfile | str,
        *,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        self._http = httpx.Client(
            base_url=self.profile.base_url,
            headers=_headers(self.profile),
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def models(self) -> list[str]:
        response = self._http.get("/models")
        _raise_for_status(response)
        return [m["id"] for m in response.json().get("data", [])]

    def health(self) -> bool:
        try:
            self.models()
        except (httpx.HTTPError, InferenceError):
            return False
        return True

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Completion:
        model = model or self.profile.default_model
        payload = build_chat_payload(
            model, messages, max_tokens=max_tokens, temperature=temperature, stream=False, **extra
        )
        start = time.perf_counter()
        response = self._http.post("/chat/completions", json=payload)
        _raise_for_status(response)
        return _completion_from_json(response.json(), model, time.perf_counter() - start)

    def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive; the generator returns the timed :class:`Completion`."""
        model = model or self.profile.default_model
        payload = build_chat_payload(
            model, messages, max_tokens=max_tokens, temperature=temperature, stream=True, **extra
        )
        start = time.perf_counter()
        acc = _StreamAccumulator(model, start)
        with self._http.stream("POST", "/chat/completions", json=payload) as response:
            if response.is_error:
                response.read()
                _raise_for_status(response)
            for chunk in parse_sse_lines(response.iter_lines()):
                delta = acc.add(chunk)
                if delta:
                    yield delta
        return acc.result()

    def timed_stream(self, messages: list[Message], **kwargs: Any) -> Completion:
        """Consume a stream fully and return its timed :class:`Completion`."""
        gen = self.stream_chat(messages, **kwargs)
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                return stop.value


class AsyncInferenceClient:
    """Async twin of :class:`InferenceClient`, used by the load generator."""

    def __init__(
        self,
        profile: EngineProfile | str,
        *,
        timeout: float = 300.0,
        max_connections: int = 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        self._http = httpx.AsyncClient(
            base_url=self.profile.base_url,
            headers=_headers(self.profile),
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections),
            transport=transport,
        )

    async def __aenter__(self) -> AsyncInferenceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream_deltas(
        self, messages: list[Message], *, model: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        model = model or self.profile.default_model
        max_tokens = kwargs.pop("max_tokens", 256)
        temperature = kwargs.pop("temperature", 0.0)
        payload = build_chat_payload(
            model, messages, max_tokens=max_tokens, temperature=temperature, stream=True, **kwargs
        )
        async with self._http.stream("POST", "/chat/completions", json=payload) as response:
            if response.is_error:
                await response.aread()
                _raise_for_status(response)
            async for line in response.aiter_lines():
                for chunk in parse_sse_lines([line]):
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text

    async def timed_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Completion:
        model = model or self.profile.default_model
        payload = build_chat_payload(
            model, messages, max_tokens=max_tokens, temperature=temperature, stream=True, **extra
        )
        start = time.perf_counter()
        acc = _StreamAccumulator(model, start)
        async with self._http.stream("POST", "/chat/completions", json=payload) as response:
            if response.is_error:
                await response.aread()
                _raise_for_status(response)
            async for line in response.aiter_lines():
                for chunk in parse_sse_lines([line]):
                    acc.add(chunk)
        return acc.result()
