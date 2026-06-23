"""Kernel fusion and layer fusion, the other half of TensorRT-LLM's speed.

**Kernel fusion.** In eager PyTorch, ``gelu(x @ W + b) + residual`` launches separate GPU
kernels for the matmul, the bias add, the GELU and the residual add. Every elementwise
kernel reads its input tensor from HBM and writes its output back, so the activations
make several round trips through slow memory, and each launch costs a few microseconds.
A fused kernel applies bias, GELU and residual in the matmul's epilogue while the tile is
still in registers: one read, one write, one launch.

**Layer fusion.** Q, K and V projections read the same input. Concatenating their
weights turns three GEMMs into one larger, more efficient GEMM - same math, fewer
launches, better tensor-core utilization.

TensorRT does both automatically when it compiles the engine. Below, numpy shows that
the fused forms compute the same values, and a traffic model shows why they are faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def unfused_mlp_epilogue(x: np.ndarray, w: np.ndarray, b: np.ndarray, residual: np.ndarray):
    """Four 'kernels'; every intermediate is materialized (written to and re-read from HBM)."""
    h1 = x @ w
    h2 = h1 + b
    h3 = gelu(h2)
    return h3 + residual


def fused_mlp_epilogue(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, residual: np.ndarray, tile: int = 64
) -> np.ndarray:
    """One 'kernel': compute a tile of the matmul and finish it before moving on.

    Each output tile is produced, biased, activated and added to the residual while it is
    'on chip' (a local variable), then written once.
    """
    out = np.empty((x.shape[0], w.shape[1]), dtype=np.result_type(x, w))
    for start in range(0, w.shape[1], tile):
        cols = slice(start, start + tile)
        acc = x @ w[:, cols]
        out[:, cols] = gelu(acc + b[cols]) + residual[:, cols]
    return out


def fuse_qkv(wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> np.ndarray:
    return np.concatenate([wq, wk, wv], axis=1)


def split_qkv(qkv: np.ndarray, q_dim: int, kv_dim: int):
    return qkv[:, :q_dim], qkv[:, q_dim : q_dim + kv_dim], qkv[:, q_dim + kv_dim :]


@dataclass(frozen=True)
class TrafficEstimate:
    kernel_launches: int
    activation_bytes: int
    time_us: float


def epilogue_traffic(
    tokens: int,
    hidden: int,
    *,
    elementwise_ops: int = 3,
    dtype_bytes: int = 2,
    fused: bool,
    hbm_gb_s: float = 3350.0,
    launch_overhead_us: float = 4.0,
) -> TrafficEstimate:
    """HBM traffic and time of the epilogue after a matmul producing ``[tokens, hidden]``.

    Unfused: the matmul writes its output, then each elementwise op reads + writes it, and
    the residual add also reads the residual. Fused: only the residual read and final write.
    ``hbm_gb_s`` defaults to an H100 SXM.
    """
    tensor = tokens * hidden * dtype_bytes
    if fused:
        launches = 1
        traffic = tensor * 2  # read residual, write result
    else:
        launches = 1 + elementwise_ops
        traffic = tensor + elementwise_ops * 2 * tensor + tensor  # + residual read
    seconds = traffic / (hbm_gb_s * 1e9)
    return TrafficEstimate(launches, traffic, seconds * 1e6 + launches * launch_overhead_us)


def fusion_report(tokens: int = 256, hidden: int = 14336, layers: int = 32) -> dict[str, float]:
    """Per-forward-pass savings of fusing the MLP epilogue in every layer (Llama-3-8B shape)."""
    unfused = epilogue_traffic(tokens, hidden, fused=False)
    fused = epilogue_traffic(tokens, hidden, fused=True)
    return {
        "launches_saved": (unfused.kernel_launches - fused.kernel_launches) * layers,
        "traffic_reduction_x": round(unfused.activation_bytes / fused.activation_bytes, 2),
        "unfused_us": round(unfused.time_us * layers, 1),
        "fused_us": round(fused.time_us * layers, 1),
    }
