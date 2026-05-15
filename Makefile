.PHONY: help venv install up down restart ps pull-model run \
        ingest ingest-sync ingest-historical reset-collection reset-and-historical \
        stats health logs clean

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
API ?= http://localhost:8000/api/v1
DAYS ?= 7

help:
	@echo "Available targets:"
	@echo "  make venv                  - Create virtual environment"
	@echo "  make install               - Install Python dependencies"
	@echo "  make up                    - Start Docker services (Ollama + ChromaDB)"
	@echo "  make down                  - Stop Docker services"
	@echo "  make restart               - Restart Docker services"
	@echo "  make ps                    - Show Docker service status"
	@echo "  make pull-model            - Pull default Ollama model in container"
	@echo "  make run                   - Start FastAPI app with reload"
	@echo ""
	@echo "  make ingest                - Trigger today's ingestion (async, returns immediately)"
	@echo "  make ingest-sync           - Trigger today's ingestion (wait for result)"
	@echo "  make ingest-historical     - Ingest last DAYS days (default: DAYS=7)"
	@echo "  make reset-collection      - Wipe the ChromaDB collection"
	@echo "  make reset-and-historical  - Wipe + ingest last DAYS days (default: DAYS=7)"
	@echo ""
	@echo "  make stats                 - Show API stats (chunk count, collection)"
	@echo "  make health                - Show API health (Ollama + ChromaDB)"
	@echo "  make logs                  - Tail application logs"
	@echo "  make clean                 - Remove Python cache files"

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
	curl -s -X POST $(API)/ingest/trigger | python3 -m json.tool

ingest-sync:
	curl -s -X POST $(API)/ingest/trigger/sync | python3 -m json.tool

ingest-historical:
	curl -s -X POST $(API)/ingest/historical \
	     -H "Content-Type: application/json" \
	     -d '{"days": $(DAYS)}' | python3 -m json.tool

reset-collection:
	curl -s -X POST $(API)/collection/reset | python3 -m json.tool

reset-and-historical:
	curl -s -X POST $(API)/ingest/reset-and-historical \
	     -H "Content-Type: application/json" \
	     -d '{"days": $(DAYS)}' | python3 -m json.tool

stats:
	curl -s $(API)/stats | python3 -m json.tool

health:
	curl -s $(API)/health | python3 -m json.tool

logs:
	tail -f logs/app.log

clean:
	rm -rf __pycache__ app/__pycache__ scheduler/__pycache__
