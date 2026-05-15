# stock-rag-bot

Local-LLM-powered RAG bot for Turkish capital markets (BIST). Scrapes daily bulletins from Ziraat Yatırım, embeds them with `BAAI/bge-m3`, stores them in ChromaDB, and answers natural-language questions via Ollama.

---

## Architecture

```
Telegram / REST API
        │
        ▼
  RAG Chain (chain.py)
  ├── Retriever  →  ChromaDB (localhost:8001)  ←  SentenceTransformers (BAAI/bge-m3)
  └── Generator  →  Ollama (localhost:11434)   ←  qwen2.5:7b (or any pulled model)
        │           JSON mode enforced + 1-retry on parse failure
        ▲
  Ingestion Pipeline
  └── ZiraatYatirimScraper
      └── Sabah Stratejisi  (HTML → semantic records per company, daily)
```

---

## Prerequisites

- Docker + Docker Compose v2
- NVIDIA Container Toolkit (for GPU in Docker)
- Python 3.11+
- An NVIDIA GPU (tested on RTX 5070)

---

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env — at minimum set TELEGRAM_BOT_TOKEN
```

Key `.env` variables:

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** From @BotFather |
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM used for generation |
| `CHROMA_COLLECTION_NAME` | `financial_bulletins` | Collection name in ChromaDB |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | SentenceTransformers model |
| `EMBEDDING_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `RETRIEVER_TOP_K` | `6` | Chunks returned per query |
| `RAG_MIN_CHUNK_SCORE` | `0.12` | Minimum cosine similarity to use a chunk |
| `INGEST_CRON_HOUR` | `8` | Hour for the daily auto-ingestion job |

**LLM generation settings** (not in `.env` — change in `app/rag/chain.py`):

| Constant | Value | Description |
|---|---|---|
| `_NUM_PREDICT_DEFAULT` | `1024` | Token budget for single-stock queries |
| `_NUM_PREDICT_OVERVIEW` | `2500` | Token budget for morning overview (up to 18 companies) |

The LLM is called with Ollama's **JSON mode** (`format="json"`), which constrains the model at the inference level to produce syntactically valid JSON. If parsing still fails, the chain retries once with an explicit reminder before returning a degraded response.

### 2. Start infrastructure

```bash
make up          # starts Ollama + ChromaDB containers
make pull-model  # pulls qwen2.5:7b into the Ollama container
```

### 3. Install Python dependencies

```bash
make venv
make install
```

### 4. Run the app

```bash
make run
# → FastAPI on http://localhost:8000
# → Telegram bot polling starts automatically
# → Daily ingestion scheduler starts (weekdays at INGEST_CRON_HOUR)
```

---

## First-time ingestion

On a fresh database, run ingestion before querying:

```bash
make ingest-sync
```

The scheduler then runs automatically every weekday at `INGEST_CRON_HOUR` (default 08:00) to keep the collection up to date with each day's bulletin.

**Clean start** — wipe the database and re-ingest today's bulletin:
```bash
make reset-collection && make ingest-sync
```

---

## Telegram Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message + quick-access keyboard |
| `/analiz THYAO` | Daily analysis for a specific BIST ticker |
| `/haftalik GARAN` | Weekly multi-source summary for a ticker |
| `/kurumlar` | Select a brokerage source filter |
| `/model` | Switch the LLM model for your session |
| `/ingest` | Manually trigger today's ingestion (admin) |
| `/durum` | System health: Ollama status + chunk count |

**Free-text queries** (type directly without a command):

```
Bugün sabah stratejisinde hangi şirketler var?
THYAO hissesinde bu hafta neler var?
Bugün ne var?
Sabah bülteninde neler var?
EREGL
```

A message that looks like a bare ticker (`THYAO`, `GARAN`, etc.) opens the analysis type keyboard directly.

---

## REST API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs`

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Ollama + ChromaDB connectivity check |
| `GET` | `/stats` | Collection name and document count |
| `POST` | `/collection/reset` | **Wipe the entire ChromaDB collection.** Returns `previous_count`. |

**Reset collection example:**
```bash
curl -X POST http://localhost:8000/api/v1/collection/reset
# {"success":true,"message":"Collection wiped...","previous_count":312}
```

---

### RAG Queries

#### `POST /query` — Single-stock analysis

```json
{
  "query": "THYAO için bu haftaki görünüm nedir?",
  "stock_code": "THYAO",
  "source": "ziraat_yatirim",
  "days_back": 7,
  "model": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Natural-language question (3–500 chars) |
| `stock_code` | string\|null | null | BIST ticker filter |
| `source` | string\|null | null | Brokerage filter (`ziraat_yatirim`) |
| `days_back` | int | 7 | Search window in days (1–90) |
| `model` | string\|null | null | Ollama model override |

#### `POST /query/weekly` — Weekly synthesis

```json
{
  "stock_code": "GARAN",
  "model": null
}
```

#### `POST /query/free` — Free-form RAG query

```json
{
  "query": "Bugün sabah bülteninde hangi şirketler öne çıktı?",
  "days_back": 14,
  "source": null,
  "model": null
}
```

**Response shape (all query endpoints):**
```json
{
  "success": true,
  "query": "...",
  "result": {
    "ozet": "...",
    "hisse_kodu": "THYAO",
    "sirket_haber_ozetleri": [],
    "kaynaklar": ["ziraat_yatirim - 2026-05-15"],
    "onemli_notlar": [],
    "yeterli_veri": true
  },
  "sources": [{"source": "ziraat_yatirim", "date": "2026-05-15", "url": "...", "stock_code": "THYAO"}],
  "model_used": "qwen2.5:7b",
  "chunks_retrieved": 4
}
```

---

### Ingestion

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/ingest/trigger` | — | Start today's ingestion in the background |
| `POST` | `/ingest/trigger/sync` | — | Run today's ingestion and wait for result |

Ingestion is idempotent — running it twice on the same day upserts the same chunk IDs and produces no duplicates.

**Response:**
```json
{
  "success": true,
  "message": "Ingestion complete",
  "total_documents": 12,
  "total_chunks": 12,
  "upserted": 12,
  "errors": []
}
```

---

## Makefile Reference

```bash
make help                 # list all targets
make venv                 # create .venv
make install              # pip install -r requirements.txt
make up                   # docker compose up -d
make down                 # docker compose down
make restart              # docker compose restart
make ps                   # docker compose ps
make pull-model           # pull qwen2.5:7b into the Ollama container
make run                  # uvicorn with --reload

# Ingestion
make ingest               # async trigger (returns immediately)
make ingest-sync          # sync trigger (waits for result)

# Monitoring
make stats                # GET /api/v1/stats
make health               # GET /api/v1/health
make logs                 # tail -f logs/app.log

make clean                # remove __pycache__ directories
```

---

## Adding a New Scraper

1. Create `app/ingestion/scrapers/<name>.py` subclassing `BaseScraper`.
2. Implement `fetch_bulletins() -> list[BulletinDocument]`.
3. Add the class to `ACTIVE_SCRAPERS` in `app/ingestion/pipeline.py`.
4. Add a `"source_key": "Human Name"` entry to `KURUM_BY_SOURCE` in `app/ingestion/scrapers/semantic.py`.

---

## Troubleshooting

**"Yeterli veri bulunamadı" on every query**

Check whether the collection has data:
```bash
make stats
# or
make health
```
If `document_count` is 0, run `make ingest-sync`.

**Date filter removing all results**

Set `LOG_LEVEL=DEBUG` in `.env` and restart. Look for:
```
[retriever] All candidates removed by date filter ... Dates seen in collection: [...]
```
The log shows the actual dates stored. If they are outside the query window, increase `RETRIEVER_TOP_K` days or re-ingest with a wider window.

**Embedding dimension mismatch after changing `EMBEDDING_MODEL`**

The existing ChromaDB collection was created with a different vector dimension. Run:
```bash
make reset-collection
make ingest-sync
```

**LLM returns empty or malformed response on overview queries**

The morning overview (up to 18 companies) uses `_NUM_PREDICT_OVERVIEW = 2500` tokens. If
responses still look cut off, increase this constant in `app/rag/chain.py`. The chain
automatically retries once when the LLM output cannot be parsed as JSON — check the logs
for `[rag] JSON parse failed — retrying once`.

**Ollama not reachable**

```bash
make health   # check ollama_connected field
make ps       # verify container is running
docker logs ollama
```
