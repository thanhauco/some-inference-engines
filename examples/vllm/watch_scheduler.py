"""Poll vLLM's Prometheus metrics to see the scheduler while you load-test it.

    python examples/vllm/watch_scheduler.py http://localhost:8000
    ie bench vllm --concurrency 64 --requests 512        # in another terminal

`waiting > 0` with `kv_cache_usage` near 100% means memory, not compute, is the limit.
"""

import sys
import time

import httpx

from inference_engines.engines.vllm import scheduler_snapshot

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/") + "/metrics"
while True:
    snap = scheduler_snapshot(httpx.get(url, timeout=5).text)
    usage = snap["kv_cache_usage"]
    print(
        f"running={snap['running']!s:>6}  waiting={snap['waiting']!s:>6}  "
        f"kv_cache={'n/a' if usage is None else f'{usage:.0%}'}"
    )
    time.sleep(1)
