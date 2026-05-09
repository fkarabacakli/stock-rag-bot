from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str = "CHANGE_ME"

    # Ollama (Docker container — localhost:11434, RTX 5070)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Optional vision model (for Technical Bulletin images, e.g. llava:7b)
    ollama_vision_model: str = ""
    ollama_timeout: int = 120

    # ChromaDB (Docker HTTP server — localhost:8001)
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "financial_bulletins"

    # Embedding (host'ta SentenceTransformers + PyTorch)
    embedding_model: str = "BAAI/bge-m3"
    # auto: use GPU if CUDA is available, otherwise CPU | cuda | cpu — requires a CUDA-enabled PyTorch build
    embedding_device: str = "auto"

    # RAG
    retriever_top_k: int = 6
    retriever_mmr_lambda: float = 0.6
    # Broad Sabah Stratejisi questions (e.g. "which companies") — more chunks with slightly higher diversity
    rag_sabah_overview_top_k: int = 20
    rag_sabah_overview_mmr_lambda: float = 0.42
    rag_min_chunk_score: float = 0.12

    # Scheduler
    ingest_cron_hour: int = 8
    ingest_cron_minute: int = 0

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Scraper
    scraper_request_timeout: int = 30
    scraper_retry_attempts: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
