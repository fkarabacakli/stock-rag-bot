.PHONY: help venv install up down restart ps pull-model run ingest ingest-sync stats health logs clean

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install      - Install Python dependencies"
	@echo "  make up           - Start Docker services (Ollama + Chroma)"
	@echo "  make down         - Stop Docker services"
	@echo "  make restart      - Restart Docker services"
	@echo "  make ps           - Show Docker service status"
	@echo "  make pull-model   - Pull default Ollama model in container"
	@echo "  make run          - Start FastAPI app with reload"
	@echo "  make ingest       - Trigger async ingestion"
	@echo "  make ingest-sync  - Trigger sync ingestion"
	@echo "  make stats        - Show API stats"
	@echo "  make health       - Show API health"
	@echo "  make logs         - Tail application logs"
	@echo "  make clean        - Remove Python cache files"

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install:
	$(VENV_PIP) install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

ps:
	docker compose ps

pull-model:
	docker exec -it ollama ollama pull qwen2.5:7b

run:
	$(VENV_PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	curl -X POST http://localhost:8000/api/v1/ingest/trigger

ingest-sync:
	curl -X POST http://localhost:8000/api/v1/ingest/trigger/sync

stats:
	curl http://localhost:8000/api/v1/stats

health:
	curl http://localhost:8000/api/v1/health

logs:
	tail -f logs/app.log

clean:
	rm -rf __pycache__ app/__pycache__ scheduler/__pycache__
