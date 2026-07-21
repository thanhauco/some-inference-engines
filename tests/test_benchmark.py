import asyncio
import math

import httpx
import pytest
from fake_engine import handler

from inference_engines.benchmark.loadgen import DEFAULT_PROMPTS, LoadConfig, run_load, sweep
from inference_engines.benchmark.metrics import RequestRecord, goodput, percentile, summarize
from inference_engines.benchmark.report import to_json, to_markdown
from inference_engines.client import AsyncInferenceClient
from inference_engines.config import get_profile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile(list(range(101)), 99) == 99
    assert math.isnan(percentile([], 50))


def test_summary_and_goodput():
    records = [
        RequestRecord(0.0, 1.0, 0.1, 10),
        RequestRecord(0.5, 1.5, 0.5, 11),
        RequestRecord(1.0, 0.2, None, 0, error="timeout"),
    ]
    s = summarize(records)
    assert s.duration_s == 2.0 and s.errors == 1
    assert s.output_token_throughput == pytest.approx(21 / 2.0)
    assert s.ttft_ms["p50"] == pytest.approx(300.0)
    assert s.tpot_ms["mean"] == pytest.approx(1000 * (0.1 + 0.1) / 2)
    assert goodput(records, 2.0, ttft_slo_s=0.2, tpot_slo_s=0.2) == 0.5


def _client(h=handler):
    return AsyncInferenceClient(get_profile("vllm", env={}), transport=httpx.MockTransport(h))


def test_closed_loop_run_against_fake_engine():
    async def go():
        async with _client() as client:
            return await run_load(
                client, DEFAULT_PROMPTS, LoadConfig(num_requests=12, concurrency=4)
            )

    result = asyncio.run(go())
    assert result.summary.requests == 12 and result.summary.errors == 0
    assert all(r.output_tokens > 0 for r in result.records)
    assert "| vllm | 4 | - |" in to_markdown([result])
    assert '"engine": "vllm"' in to_json([result])


def test_open_loop_and_errors_are_recorded():
    def flaky(request):
        return httpx.Response(503, text="overloaded")

    async def go():
        async with _client(flaky) as client:
            return await run_load(
                client, DEFAULT_PROMPTS, LoadConfig(num_requests=5, concurrency=2, request_rate=500)
            )

    result = asyncio.run(go())
    assert result.summary.errors == 5
    assert "503" in result.records[0].error


def test_sweep_covers_each_level():
    async def go():
        async with _client() as client:
            return await sweep(client, DEFAULT_PROMPTS, concurrencies=(1, 3), requests_per_level=3)

    assert [r.config.concurrency for r in asyncio.run(go())] == [1, 3]
