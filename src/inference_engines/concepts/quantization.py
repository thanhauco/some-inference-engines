"""Quantization: fewer bits per weight, the cheapest speedup TensorRT-LLM (and others) offer.

Decoding is memory-bandwidth bound, so the time per token is roughly
``bytes of weights / HBM bandwidth``. Halving the bytes nearly halves the time, and frees
VRAM for a bigger KV cache (= larger batches). The question is how much accuracy you pay.

* **INT8 per-channel** (weight-only, W8A16): one scale per output channel.
* **INT4 group-wise** (W4A16, the AWQ/GPTQ format): one scale per group of 128 inputs;
  smaller groups track outliers better at the cost of more scales.
* **FP8 E4M3** (W8A8 on Hopper/Blackwell): a floating-point byte - 4 exponent bits,
  3 mantissa bits, max 448. Tensor cores multiply it natively, so activations are
  quantized too and the matmul itself runs at twice the BF16 rate.

Everything here is numpy emulation, to compare *errors* and *sizes*; real engines use
calibrated scales (and tricks like SmoothQuant/AWQ) and fused low-precision kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FP8_E4M3_MAX = 448.0


@dataclass
class Quantized:
    values: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int | None = None

    def dequantize(self) -> np.ndarray:
        if self.group_size is None:
            return self.values.astype(np.float32) * self.scales
        out_dim, in_dim = self.values.shape
        grouped = self.values.reshape(out_dim, in_dim // self.group_size, self.group_size)
        return (grouped.astype(np.float32) * self.scales[..., None]).reshape(out_dim, in_dim)

    @property
    def nbytes(self) -> float:
        """Storage in bytes: packed low-bit values plus fp16 scales."""
        return self.values.size * self.bits / 8 + self.scales.size * 2


def quantize_int8_per_channel(w: np.ndarray) -> Quantized:
    """Symmetric INT8 with one scale per output row of ``w`` (shape ``[out, in]``)."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    scales = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(w / scales), -127, 127).astype(np.int8)
    return Quantized(q, scales, bits=8)


def quantize_int4_groupwise(w: np.ndarray, group_size: int = 128) -> Quantized:
    """Symmetric INT4 (values in [-8, 7]) with one scale per ``group_size`` input columns."""
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"in_dim {in_dim} is not divisible by group_size {group_size}")
    grouped = w.reshape(out_dim, in_dim // group_size, group_size)
    amax = np.abs(grouped).max(axis=2)
    scales = np.where(amax == 0, 1.0, amax / 7.0).astype(np.float32)
    q = np.clip(np.round(grouped / scales[..., None]), -8, 7).astype(np.int8)
    return Quantized(q.reshape(out_dim, in_dim), scales, bits=4, group_size=group_size)


def round_to_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest representable FP8 E4M3 value (saturating)."""
    x = np.asarray(x, dtype=np.float32)
    mag = np.minimum(np.abs(x), FP8_E4M3_MAX)
    # Exponent of each value; below 2^-6 the format is subnormal with a fixed 2^-9 step.
    exp = np.floor(np.log2(np.where(mag > 0, mag, 1.0)))
    exp = np.maximum(exp, -6)
    step = np.exp2(exp - 3)  # 3 mantissa bits
    rounded = np.round(mag / step) * step
    return (np.sign(x) * np.minimum(rounded, FP8_E4M3_MAX)).astype(np.float32)


def quantize_fp8(w: np.ndarray) -> Quantized:
    """Per-tensor scaled FP8 E4M3: scale so the largest |w| maps to 448, then round."""
    amax = float(np.abs(w).max()) or 1.0
    scale = np.float32(amax / FP8_E4M3_MAX)
    return Quantized(round_to_fp8_e4m3(w / scale), np.array(scale), bits=8)


def relative_error(reference: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(reference - approx) / np.linalg.norm(reference))


def compare_formats(
    out_dim: int = 1024,
    in_dim: int = 1024,
    tokens: int = 16,
    outlier_fraction: float = 0.001,
    seed: int = 0,
) -> list[dict[str, float | str]]:
    """Quantize one synthetic linear layer every way and measure output error and size.

    LLM weights are roughly Gaussian with rare large outliers; the outliers are what make
    coarse per-tensor or per-channel scales lose precision.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(out_dim, in_dim)).astype(np.float32)
    mask = rng.random(w.shape) < outlier_fraction
    w[mask] *= 20
    x = rng.normal(size=(in_dim, tokens)).astype(np.float32)
    reference = w @ x
    fp16_bytes = w.size * 2
    rows: list[dict[str, float | str]] = [
        {"format": "FP16 (baseline)", "bytes": fp16_bytes, "compression": 1.0, "rel_error": 0.0}
    ]
    candidates = {
        "INT8 per-channel (W8A16)": quantize_int8_per_channel(w),
        "FP8 E4M3 per-tensor": quantize_fp8(w),
        "INT4 group-128 (W4A16)": quantize_int4_groupwise(w, 128),
        "INT4 group-32": quantize_int4_groupwise(w, 32),
    }
    for name, q in candidates.items():
        rows.append(
            {
                "format": name,
                "bytes": int(q.nbytes),
                "compression": round(fp16_bytes / q.nbytes, 2),
                "rel_error": round(relative_error(reference, q.dequantize() @ x), 5),
            }
        )
    return rows


def decode_speedup_estimate(bits: float, baseline_bits: float = 16.0) -> float:
    """Upper bound on decode speedup when weight reads dominate step time."""
    return baseline_bits / bits
