"""Talk to a running vLLM server with the official OpenAI SDK - no vLLM import needed.

./examples/vllm/serve.sh            # in another terminal
pip install openai && python examples/vllm/openai_client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = client.models.list().data[0].id

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Why does continuous batching raise throughput?"}],
    max_tokens=300,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
