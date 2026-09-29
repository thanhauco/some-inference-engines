"""The same application on four engines.

"You may not stay with the same engine forever." This app never imports an engine: it
picks one from the ENGINE variable and talks the OpenAI API. Moving from Ollama on a
laptop to vLLM, SGLang or TensorRT-LLM in production is a config change.

    ENGINE=ollama        python examples/migration/app.py
    ENGINE=vllm          python examples/migration/app.py
    ENGINE=sglang        python examples/migration/app.py
    ENGINE=tensorrt-llm  python examples/migration/app.py
"""

from __future__ import annotations

import os
import sys

from inference_engines.client import InferenceClient, InferenceError
from inference_engines.config import get_profile

SYSTEM = "You are a release-notes assistant. Answer in three bullet points."


def summarize(client: InferenceClient, changelog: str) -> str:
    return client.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Summarize for customers:\n{changelog}"},
        ],
        max_tokens=200,
    ).text


def main() -> int:
    profile = get_profile(os.environ.get("ENGINE", "ollama"))
    print(f"engine: {profile.engine.value}  ({profile.tagline})")
    print(f"model:  {profile.default_model}\nurl:    {profile.base_url}\n")
    with InferenceClient(profile) as client:
        if not client.health():
            print(
                f"{profile.engine.value} is not reachable. Start it with:\n  {profile.launch_hint}",
                file=sys.stderr,
            )
            return 1
        try:
            print(
                summarize(client, "- Added FP8 KV cache\n- p99 TTFT down 40%\n- Fixed OOM at 128k")
            )
        except InferenceError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
