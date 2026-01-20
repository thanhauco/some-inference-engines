"""Engine profiles: where each engine listens and which model it serves by default.

All four engines expose an OpenAI-compatible HTTP API, which is what makes switching
engines cheap. Defaults use the *same* model family (Llama 3.1 8B Instruct) everywhere so
that a migration changes the engine, not the model.

Every field can be overridden with environment variables, e.g.::

    IE_VLLM_BASE_URL=http://gpu-box:8000/v1
    IE_OLLAMA_MODEL=llama3.1:8b-instruct-q8_0
    IE_TENSORRT_LLM_API_KEY=secret
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum


class Engine(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt-llm"

    @classmethod
    def parse(cls, value: str | Engine) -> Engine:
        if isinstance(value, Engine):
            return value
        normalized = value.strip().lower().replace("_", "-")
        aliases = {"trt": "tensorrt-llm", "trtllm": "tensorrt-llm", "trt-llm": "tensorrt-llm"}
        return cls(aliases.get(normalized, normalized))


@dataclass(frozen=True)
class EngineProfile:
    engine: Engine
    base_url: str
    """OpenAI-compatible root, including the ``/v1`` suffix."""
    default_model: str
    tagline: str
    launch_hint: str
    api_key: str = "EMPTY"

    @property
    def root_url(self) -> str:
        """Server root without the ``/v1`` suffix (for native, non-OpenAI endpoints)."""
        url = self.base_url.rstrip("/")
        return url[: -len("/v1")] if url.endswith("/v1") else url


DEFAULT_PROFILES: dict[Engine, EngineProfile] = {
    Engine.OLLAMA: EngineProfile(
        engine=Engine.OLLAMA,
        base_url="http://localhost:11434/v1",
        default_model="llama3.1:8b",
        tagline="Simple, local LLM running on your machine",
        launch_hint="ollama serve & ollama pull llama3.1:8b",
    ),
    Engine.VLLM: EngineProfile(
        engine=Engine.VLLM,
        base_url="http://localhost:8000/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        tagline="High-throughput serving for production LLMs",
        launch_hint="vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000",
    ),
    Engine.SGLANG: EngineProfile(
        engine=Engine.SGLANG,
        base_url="http://localhost:30000/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        tagline="Flexible, high-performance serving with programmable control",
        launch_hint=(
            "python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct "
            "--port 30000"
        ),
    ),
    Engine.TENSORRT_LLM: EngineProfile(
        engine=Engine.TENSORRT_LLM,
        # trtllm-serve defaults to :8000 too; we move it to :8001 so it can run next to vLLM.
        base_url="http://localhost:8001/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        tagline="Maximum performance with NVIDIA optimizations",
        launch_hint="trtllm-serve meta-llama/Llama-3.1-8B-Instruct --port 8001",
    ),
}


def _env_prefix(engine: Engine) -> str:
    return "IE_" + engine.value.upper().replace("-", "_")


def get_profile(engine: str | Engine, env: Mapping[str, str] | None = None) -> EngineProfile:
    """Return the profile for ``engine`` with any ``IE_<ENGINE>_*`` overrides applied."""
    engine = Engine.parse(engine)
    env = os.environ if env is None else env
    profile = DEFAULT_PROFILES[engine]
    prefix = _env_prefix(engine)
    return replace(
        profile,
        base_url=env.get(f"{prefix}_BASE_URL", profile.base_url),
        default_model=env.get(f"{prefix}_MODEL", profile.default_model),
        api_key=env.get(f"{prefix}_API_KEY", profile.api_key),
    )


def all_profiles(env: Mapping[str, str] | None = None) -> list[EngineProfile]:
    return [get_profile(engine, env) for engine in Engine]
