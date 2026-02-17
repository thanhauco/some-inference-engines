#!/usr/bin/env bash
# Ollama in five commands: install, serve, pull, run, call the API.
set -euo pipefail
MODEL="${MODEL:-llama3.1:8b}"

command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh

# 1. Start the daemon (the desktop app does this for you).
if ! pgrep -x ollama >/dev/null; then
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

# 2. Download the model from the registry once (a ~4.9 GB 4-bit GGUF).
ollama pull "$MODEL"

# 3. Run it interactively... or one-shot:
ollama run "$MODEL" "In one sentence, what is a KV cache?"

# 4. Same model through the OpenAI-compatible API every other engine also speaks.
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi\"}]}"
echo

# 5. What's loaded, and where (CPU / GPU split)?
ollama ps
