"""Tensor parallelism vs replicas: how to spend N GPUs.

**Tensor parallelism (TP=n)** shards every weight matrix across n GPUs. Each decode step
reads only 1/n of the weights per GPU, so the step gets faster - but every layer now needs
two all-reduce exchanges between the GPUs, and that communication does not shrink with n.

**Replicas** run n independent copies. No communication cost, n times the throughput,
but no latency gain, and the model must fit on one GPU.

So: TP buys *latency* (and makes a too-big model fit); replicas buy *throughput per
dollar*. The crossover depends on interconnect speed, which is why TP=8 over NVLink is
routine and TP across nodes over Ethernet rarely is. All four engines that target
datacenter GPUs expose this choice (``--tensor-parallel-size``, ``--tp``, ``--tp_size``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .batching import CostModel
from .memory import MODELS, GiB, ModelSpec, weight_bytes


@dataclass(frozen=True)
class Interconnect:
    name: str
    allreduce_ms: float
    """Latency of one all-reduce of an activation tensor between the TP ranks."""


NVLINK = Interconnect("NVLink", 0.015)
PCIE = Interconnect("PCIe (no NVLink)", 0.08)
ETHERNET = Interconnect("multi-node RoCE/Ethernet", 0.30)


@dataclass
class ParallelPlan:
    layout: str
    gpus: int
    tp: int
    replicas: int
    fits: bool
    tpot_ms: float
    throughput_tok_s: float

    def row(self) -> dict[str, float | str | bool]:
        return {
            "layout": self.layout,
            "fits": self.fits,
            "TPOT ms": round(self.tpot_ms, 2) if self.fits else "-",
            "total tok/s": round(self.throughput_tok_s) if self.fits else "-",
            "tok/s per GPU": round(self.throughput_tok_s / self.gpus) if self.fits else "-",
        }


def step_ms(
    tp: int,
    batch: int,
    *,
    layers: int = 32,
    link: Interconnect = NVLINK,
    cost: CostModel | None = None,
) -> float:
    """Decode step time with the weights sharded over ``tp`` GPUs."""
    cost = cost or CostModel()
    compute = (cost.decode_base_ms + cost.decode_per_seq_ms * batch) / tp
    comm = 0.0 if tp == 1 else layers * 2 * link.allreduce_ms
    return compute + comm


def plan_layouts(
    model: ModelSpec | str = "llama-3.1-8b",
    *,
    gpus: int = 4,
    vram_gib: float = 80,
    quant: str = "bf16",
    batch_per_replica: int = 32,
    link: Interconnect = NVLINK,
    cost: CostModel | None = None,
) -> list[ParallelPlan]:
    """Every way to split ``gpus`` into replicas x TP, with latency and throughput.

    The cost model's ``decode_base_ms`` is scaled by the model's weight bytes relative to
    Llama-3.1-8B in BF16, since a decode step is dominated by reading the weights.
    """
    spec = MODELS[model] if isinstance(model, str) else model
    base = cost or CostModel()
    scale = weight_bytes(spec, quant) / weight_bytes(MODELS["llama-3.1-8b"], "bf16")
    scaled = CostModel(
        base.decode_base_ms * scale,
        base.decode_per_seq_ms * scale,
        base.prefill_per_token_ms * scale,
    )
    plans = []
    tp = 1
    while tp <= gpus:
        if gpus % tp == 0:
            replicas = gpus // tp
            # Leave 15% of VRAM for KV cache and runtime.
            fits = weight_bytes(spec, quant) / GiB / tp <= vram_gib * 0.85
            t = step_ms(tp, batch_per_replica, layers=spec.layers, link=link, cost=scaled)
            plans.append(
                ParallelPlan(
                    layout=f"{replicas} replica(s) x TP={tp}",
                    gpus=gpus,
                    tp=tp,
                    replicas=replicas,
                    fits=fits,
                    tpot_ms=t,
                    throughput_tok_s=replicas * batch_per_replica / (t / 1000) if fits else 0.0,
                )
            )
        tp *= 2
    return plans


def tp_speedup(tp: int, batch: int = 1, **kwargs) -> float:
    return step_ms(1, batch, **kwargs) / step_ms(tp, batch, **kwargs)
