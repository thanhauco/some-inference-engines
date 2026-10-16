.PHONY: install test lint format simulate up-ollama up-vllm up-sglang up-trtllm down

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff check --fix src tests examples
	ruff format src tests examples

simulate:
	ie simulate all

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
