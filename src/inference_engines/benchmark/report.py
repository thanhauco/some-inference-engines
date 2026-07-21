"""Render benchmark results as a Markdown table or JSON, for terminals and PRs."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .loadgen import BenchmarkResult

COLUMNS = (
    "engine",
    "conc",
    "rate",
    "req/s",
    "out tok/s",
    "TTFT p50",
    "TTFT p99",
    "TPOT p50",
    "E2E p99",
    "errors",
)


def rows(results: Sequence[BenchmarkResult]) -> list[list[str]]:
    out = []
    for r in results:
        s = r.summary
        out.append([
            r.engine,
            str(r.config.concurrency),
            f"{r.config.request_rate:g}" if r.config.request_rate else "-",
            f"{s.request_throughput:.2f}",
            f"{s.output_token_throughput:.1f}",
            f"{s.ttft_ms['p50']:.0f} ms",
            f"{s.ttft_ms['p99']:.0f} ms",
            f"{s.tpot_ms['p50']:.1f} ms",
            f"{s.e2e_ms['p99']:.0f} ms",
            str(s.errors),
        ])  # fmt: skip
    return out


def to_markdown(results: Sequence[BenchmarkResult]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row) + " |" for row in rows(results)]
    return "\n".join(lines)


def to_json(results: Sequence[BenchmarkResult]) -> str:
    return json.dumps(
        [
            {
                "engine": r.engine,
                "concurrency": r.config.concurrency,
                "request_rate": r.config.request_rate,
                **r.summary.as_dict(),
            }
            for r in results
        ],
        indent=2,
    )
