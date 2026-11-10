.PHONY: install test lint format simulate videos up-ollama up-vllm up-sglang up-trtllm down

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests examples videos
	ruff format --check src tests examples videos

format:
	ruff check --fix src tests examples videos
	ruff format src tests examples videos

simulate:
	ie simulate all

# Needs node + playwright (npm i playwright) and ffmpeg.
videos:
	python videos/export_data.py
	node videos/render.js

up-ollama:
	docker compose -f deploy/docker-compose.yml --profile ollama up -d
up-vllm:
	docker compose -f deploy/docker-compose.yml --profile vllm up -d
up-sglang:
	docker compose -f deploy/docker-compose.yml --profile sglang up -d
up-trtllm:
	docker compose -f deploy/docker-compose.yml --profile tensorrt-llm up -d
down:
	docker compose -f deploy/docker-compose.yml --profile '*' down
