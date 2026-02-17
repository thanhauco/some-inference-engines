"""Ollama's native API: make sure a model exists, generate, and read Ollama's own timings.

python examples/ollama/native_api.py [model]
"""

import sys

from inference_engines.client import InferenceClient
from inference_engines.engines.ollama import OllamaNative, tokens_per_second

model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1:8b"
native = OllamaNative()
print("ollama", native.version())

if native.ensure_model(model):
    print(f"pulled {model}")

response = native.generate(model, "List three uses of a local LLM.", temperature=0)
print(response["response"])
print(
    f"\ndecode speed: {tokens_per_second(response):.1f} tok/s "
    f"(load {response.get('load_duration', 0) / 1e9:.2f} s)"
)
print("loaded:", [(m["name"], m.get("size_vram")) for m in native.running_models()])

# The exact same model through the engine-agnostic client used by the rest of the repo.
with InferenceClient("ollama") as client:
    print(client.chat([{"role": "user", "content": "One word: sky color?"}], model=model).text)
