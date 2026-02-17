"""Ollama: the simplest path from zero to a running model.

Ollama wraps llama.cpp behind a model registry (``ollama pull``) and a small daemon
(``ollama serve`` on :11434). Models are GGUF files, usually 4-bit quantized, so an 8B
model fits on a laptop and runs on CPU, NVIDIA/AMD GPUs or Apple Silicon (Metal).

Besides the OpenAI-compatible ``/v1`` API, Ollama has a native API used here for the
things OpenAI's API has no concept of: pulling models, listing local models and seeing
what is loaded in memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from ..config import Engine, EngineProfile, get_profile


def launch_command() -> list[str]:
    return ["ollama", "serve"]


def pull_command(model: str) -> list[str]:
    return ["ollama", "pull", model]


def modelfile(
    base: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> str:
    """Render a Modelfile: Ollama's Dockerfile-like recipe for a customised model."""
    lines = [f"FROM {base}"]
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")
    if num_ctx is not None:
        lines.append(f"PARAMETER num_ctx {num_ctx}")
    if system:
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"


class OllamaNative:
    """Client for Ollama's native ``/api`` endpoints."""

    def __init__(
        self,
        profile: EngineProfile | None = None,
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.profile = profile or get_profile(Engine.OLLAMA)
        self._http = httpx.Client(base_url=self.profile.root_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def version(self) -> str:
        return self._http.get("/api/version").raise_for_status().json()["version"]

    def local_models(self) -> list[str]:
        """Models already downloaded (``ollama list``)."""
        return [m["name"] for m in self._http.get("/api/tags").raise_for_status().json()["models"]]

    def running_models(self) -> list[dict[str, Any]]:
        """Models currently loaded in RAM/VRAM (``ollama ps``)."""
        return self._http.get("/api/ps").raise_for_status().json().get("models", [])

    def pull(self, model: str) -> Iterator[dict[str, Any]]:
        """Download ``model`` from the registry, yielding progress events."""
        with self._http.stream("POST", "/api/pull", json={"model": model}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def ensure_model(self, model: str) -> bool:
        """Pull ``model`` if it is not present locally. Returns True if a download happened."""
        if any(name == model or name == f"{model}:latest" for name in self.local_models()):
            return False
        for event in self.pull(model):
            if "error" in event:
                raise RuntimeError(event["error"])
        return True

    def generate(self, model: str, prompt: str, **options: Any) -> dict[str, Any]:
        """Non-streaming ``/api/generate``. The response includes Ollama's own timing counters."""
        body = {"model": model, "prompt": prompt, "stream": False}
        if options:
            body["options"] = options
        return self._http.post("/api/generate", json=body).raise_for_status().json()


def tokens_per_second(generate_response: dict[str, Any]) -> float:
    """Decode speed from ``/api/generate`` counters (durations are in nanoseconds)."""
    count = generate_response.get("eval_count", 0)
    duration_ns = generate_response.get("eval_duration", 0)
    return count / (duration_ns / 1e9) if duration_ns else 0.0
