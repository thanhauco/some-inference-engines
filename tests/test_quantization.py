import numpy as np
import pytest

from inference_engines.concepts.quantization import (
    FP8_E4M3_MAX,
    compare_formats,
    decode_speedup_estimate,
    quantize_fp8,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
    relative_error,
    round_to_fp8_e4m3,
)

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(64, 256)).astype(np.float32)


def test_int8_round_trip_is_close():
    q = quantize_int8_per_channel(W)
    assert q.values.dtype == np.int8 and np.abs(q.values).max() <= 127
    assert relative_error(W, q.dequantize()) < 0.01


def test_int4_values_are_in_range_and_groups_help():
    q128 = quantize_int4_groupwise(W, 128)
    q32 = quantize_int4_groupwise(W, 32)
    assert q128.values.min() >= -8 and q128.values.max() <= 7
    assert relative_error(W, q32.dequantize()) < relative_error(W, q128.dequantize())
    with pytest.raises(ValueError):
        quantize_int4_groupwise(W, 100)


@pytest.mark.parametrize(
    "x, expected",
    [
        (1.0, 1.0),
        (1.1, 1.125),
        (17.0, 16.0),
        (300.0, 288.0),
        (1000.0, FP8_E4M3_MAX),
        (-3.3, -3.25),
        (2**-9, 2**-9),
        (0.0, 0.0),
    ],
)
def test_fp8_e4m3_rounding(x, expected):
    assert round_to_fp8_e4m3(np.array([x]))[0] == expected


def test_fp8_has_3_mantissa_bits_of_relative_precision():
    x = rng.uniform(0.02, 400, size=10_000).astype(np.float32)
    assert np.max(np.abs(round_to_fp8_e4m3(x) - x) / x) <= 2**-4 + 1e-6
    assert relative_error(W, quantize_fp8(W).dequantize()) < 0.05


def test_compare_formats_trades_size_for_error():
    rows = {r["format"]: r for r in compare_formats(256, 256, seed=1)}
    assert rows["INT8 per-channel (W8A16)"]["compression"] == pytest.approx(2.0, rel=0.02)
    assert rows["INT4 group-128 (W4A16)"]["compression"] > 3.5
    assert (
        rows["INT4 group-128 (W4A16)"]["rel_error"] > rows["INT8 per-channel (W8A16)"]["rel_error"]
    )
    assert decode_speedup_estimate(4) == 4.0
