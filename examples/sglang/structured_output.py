"""Schema-guaranteed JSON through the OpenAI-compatible API (works on SGLang and vLLM).

The server compiles the JSON schema to a grammar and masks invalid tokens while decoding,
so the response always parses - no retry loop.

    python examples/sglang/structured_output.py [engine]      # engine: sglang (default) | vllm
"""

import json
import sys

from inference_engines.client import InferenceClient
from inference_engines.engines.sglang import json_schema_response_format

SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "category": {"type": "string", "enum": ["billing", "performance", "bug", "other"]},
        "priority": {"type": "integer", "minimum": 1, "maximum": 4},
        "summary": {"type": "string", "maxLength": 120},
    },
    "required": ["customer", "category", "priority", "summary"],
    "additionalProperties": False,
}

TICKET = "Hi, Dana from Acme here. Since the upgrade our p99 latency tripled. Urgent!"

engine = sys.argv[1] if len(sys.argv) > 1 else "sglang"
with InferenceClient(engine) as client:
    result = client.chat(
        [{"role": "user", "content": f"Extract the ticket fields.\n\n{TICKET}"}],
        response_format=json_schema_response_format("ticket", SCHEMA),
        max_tokens=200,
    )
ticket = json.loads(result.text)  # guaranteed to parse
print(json.dumps(ticket, indent=2))
