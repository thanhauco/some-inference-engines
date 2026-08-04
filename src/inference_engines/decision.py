"""Which engine should you choose, and when?

The framework in one line each:

* **Ollama**       -> simplest local experience
* **vLLM**         -> scalable production serving
* **SGLang**       -> programmable and agentic inference
* **TensorRT-LLM** -> maximum NVIDIA performance

:func:`recommend` turns that into an explicit, explainable score. Hard constraints
(TensorRT-LLM needs an NVIDIA GPU) disqualify; everything else adds or subtracts points
with a reason attached, so you can see *why* an engine ranks where it does and argue
with the weights rather than with a benchmark chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import Engine


class Hardware(str, Enum):
    CPU = "cpu"
    APPLE_SILICON = "apple-silicon"
    NVIDIA_CONSUMER = "nvidia-consumer"
    NVIDIA_DATACENTER = "nvidia-datacenter"
    AMD_GPU = "amd-gpu"


class Workload(str, Enum):
    CHAT = "chat"
    """Prompt in, response out (assistants, completion APIs, RAG answers)."""
    AGENTIC = "agentic"
    """Multi-step programs: tool calls, branching, many calls sharing long prefixes."""
    STRUCTURED = "structured"
    """Outputs that must match a schema (JSON extraction, classification, function args)."""
    BATCH = "batch"
    """Offline throughput jobs: evals, synthetic data, bulk summarization."""


class Stage(str, Enum):
    PROTOTYPE = "prototype"
    PRODUCTION = "production"
    SCALE = "scale"


@dataclass
class Requirements:
    hardware: Hardware = Hardware.NVIDIA_DATACENTER
    workload: Workload = Workload.CHAT
    concurrency: int = 1
    """Expected simultaneous requests at peak."""
    latency_critical: bool = False
    """True when every millisecond of TTFT/TPOT has business value."""
    ops_capacity: int = 2
    """Engineering appetite for tuning: 1 = none, 2 = some, 3 = dedicated inference team."""
    local_only: bool = False
    """Must run on a developer machine or offline (privacy, air-gapped)."""
    stage: Stage = Stage.PRODUCTION


@dataclass
class Recommendation:
    engine: Engine
    score: float
    reasons: list[str] = field(default_factory=list)
    disqualified: str | None = None

    @property
    def viable(self) -> bool:
        return self.disqualified is None


def _add(rec: Recommendation, points: float, reason: str) -> None:
    rec.score += points
    sign = "+" if points >= 0 else "-"
    rec.reasons.append(f"{sign}{abs(points):.1f}  {reason}")


def _score_ollama(req: Requirements, rec: Recommendation) -> None:
    if req.hardware in (Hardware.CPU, Hardware.APPLE_SILICON):
        _add(rec, 4, "runs on CPU and Apple Silicon (llama.cpp / Metal), no CUDA needed")
    if req.local_only:
        _add(rec, 3, "one binary, local model registry, works fully offline")
    if req.stage is Stage.PROTOTYPE:
        _add(rec, 3, "fastest path to experimenting: `ollama run <model>`")
    if req.ops_capacity == 1:
        _add(rec, 2, "near-zero operational overhead")
    if req.concurrency > 8:
        _add(rec, -min(6, req.concurrency / 8), "not designed for high-concurrency serving")
    if req.stage is Stage.SCALE:
        _add(rec, -3, "lower throughput than vLLM/SGLang/TensorRT-LLM on the same GPU")
    if req.latency_critical and req.concurrency > 1:
        _add(rec, -2, "no fine control over batching or scheduling for latency targets")


def _score_vllm(req: Requirements, rec: Recommendation) -> None:
    if req.hardware in (Hardware.CPU, Hardware.APPLE_SILICON):
        _add(rec, -3, "CPU/Apple backends exist but are not where vLLM shines")
    if req.hardware in (Hardware.NVIDIA_DATACENTER, Hardware.NVIDIA_CONSUMER, Hardware.AMD_GPU):
        _add(rec, 2, "broad GPU support (NVIDIA, AMD ROCm, and more)")
    if req.concurrency > 8:
        _add(
            rec,
            min(5, 1 + req.concurrency / 32),
            "continuous batching + PagedAttention keep GPUs busy under many users",
        )
    if req.workload in (Workload.CHAT, Workload.BATCH):
        _add(rec, 2, "prompt -> response and offline batch are its home turf")
    if req.stage is Stage.PRODUCTION:
        _add(rec, 2, "mature OpenAI-compatible server, large community, wide model support")
    if req.ops_capacity >= 2:
        _add(rec, 1, "good defaults; few knobs needed to get most of the performance")
    if req.local_only and req.hardware not in (
        Hardware.NVIDIA_CONSUMER,
        Hardware.NVIDIA_DATACENTER,
    ):
        _add(rec, -2, "more setup than Ollama for a single local user")


def _score_sglang(req: Requirements, rec: Recommendation) -> None:
    if req.hardware in (Hardware.CPU, Hardware.APPLE_SILICON):
        _add(rec, -4, "built and optimized for datacenter GPUs")
    if req.hardware in (Hardware.NVIDIA_DATACENTER, Hardware.NVIDIA_CONSUMER, Hardware.AMD_GPU):
        _add(rec, 2, "high-performance GPU runtime")
    if req.workload is Workload.AGENTIC:
        _add(rec, 5, "RadixAttention reuses shared prefixes across calls; fork/select/gen programs")
    if req.workload is Workload.STRUCTURED:
        _add(rec, 4, "fast grammar-constrained decoding with jump-forward")
    if req.concurrency > 8:
        _add(
            rec, min(4, 1 + req.concurrency / 48), "continuous batching with cache-aware scheduling"
        )
    if req.ops_capacity == 1:
        _add(rec, -2, "steeper learning curve than vLLM; newer ecosystem")


def _score_trtllm(req: Requirements, rec: Recommendation) -> None:
    if req.hardware not in (Hardware.NVIDIA_DATACENTER, Hardware.NVIDIA_CONSUMER):
        rec.disqualified = "requires an NVIDIA GPU"
        return
    if req.hardware is Hardware.NVIDIA_DATACENTER:
        _add(rec, 3, "fused kernels, FP8/FP4 and in-flight batching tuned per GPU architecture")
    if req.latency_critical:
        _add(rec, 5, "lowest latency when every millisecond matters")
    if req.stage is Stage.SCALE:
        _add(rec, 3, "best GPU economics at scale (tokens per dollar)")
    if req.concurrency > 8:
        _add(rec, min(3, req.concurrency / 64), "high throughput with in-flight batching")
    if req.ops_capacity == 1:
        _add(rec, -5, "engine builds, version pinning and GPU-specific tuning need expertise")
    elif req.ops_capacity == 2:
        _add(rec, -2, "more complex build and deployment process")
    if req.workload is Workload.AGENTIC:
        _add(rec, -2, "less flexible than vLLM/SGLang for agentic workflows")
    if req.stage is Stage.PROTOTYPE:
        _add(rec, -3, "compile-and-tune overhead slows iteration")


_SCORERS = {
    Engine.OLLAMA: _score_ollama,
    Engine.VLLM: _score_vllm,
    Engine.SGLANG: _score_sglang,
    Engine.TENSORRT_LLM: _score_trtllm,
}


def recommend(req: Requirements) -> list[Recommendation]:
    """Rank all four engines for ``req``; viable engines first, best score first."""
    recs = []
    for engine, scorer in _SCORERS.items():
        rec = Recommendation(engine, 0.0)
        scorer(req, rec)
        recs.append(rec)
    return sorted(recs, key=lambda r: (not r.viable, -r.score))


MIGRATION_PATH = [
    (Engine.OLLAMA, "Experimenting", "you are choosing a model and iterating on prompts locally"),
    (
        Engine.VLLM,
        "Real users arrive",
        "concurrency, throughput and GPU utilization start to matter",
    ),
    (
        Engine.SGLANG,
        "Workflows get complex",
        "requests become programs: agents, tools, branches, schemas",
    ),
    (
        Engine.TENSORRT_LLM,
        "Latency or GPU cost is critical",
        "you will trade simplicity for NVIDIA-specific optimization",
    ),
]


def explain(recs: list[Recommendation]) -> str:
    lines = []
    for i, rec in enumerate(recs, 1):
        status = f"score {rec.score:5.1f}" if rec.viable else f"ruled out: {rec.disqualified}"
        lines.append(f"{i}. {rec.engine.value:<13} {status}")
        lines += [f"      {reason}" for reason in rec.reasons]
    return "\n".join(lines)
