"""SGLang: serving for inference *programs* - agents, tools, branching and structured output.

The runtime (SRT) adds RadixAttention, which keeps a radix tree of KV cache so any shared
prefix (system prompt, few-shot examples, the conversation so far, the trunk of a
``fork``) is computed once. The frontend language lets you write the program itself::

    @sgl.function
    def triage(s, ticket):
        s += sgl.user(ticket)
        s += sgl.assistant(sgl.select("label", choices=["bug", "billing", "other"]))

Speculative decoding (EAGLE) and grammar-constrained decoding are server flags /
request fields, see :func:`launch_command` and :func:`json_schema_response_format`.
"""

from __future__ import annotations

from typing import Any

DOCKER_IMAGE = "lmsysorg/sglang"


def launch_command(
    model: str,
    *,
    port: int = 30000,
    tp: int = 1,
    mem_fraction_static: float | None = None,
    speculative_draft_model: str | None = None,
    speculative_algorithm: str = "EAGLE",
    speculative_num_steps: int = 3,
    speculative_eagle_topk: int = 1,
    speculative_num_draft_tokens: int = 4,
    disable_radix_cache: bool = False,
) -> list[str]:
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tp", str(tp),
    ]  # fmt: skip
    if mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(mem_fraction_static)]
    if speculative_draft_model:
        cmd += [
            "--speculative-algorithm", speculative_algorithm,
            "--speculative-draft-model-path", speculative_draft_model,
            "--speculative-num-steps", str(speculative_num_steps),
            "--speculative-eagle-topk", str(speculative_eagle_topk),
            "--speculative-num-draft-tokens", str(speculative_num_draft_tokens),
        ]  # fmt: skip
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    return cmd


def json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``response_format`` that makes SGLang (and vLLM) decode under a JSON schema."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
