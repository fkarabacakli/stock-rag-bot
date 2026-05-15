"""
ChromaDB HTTP client — connects to the Docker container.

ChromaDB runs as an HTTP server on localhost:8001.
Data directory: /home/furkan/Desktop/stock-rag-bot/chroma_data (volume mount)
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.config import get_settings

_settings = get_settings()
_client = None  # chromadb ClientAPI (returns HttpClient)
_collection = None


def get_chroma_client():
    """Return a singleton connection to the Docker ChromaDB HTTP server."""
    global _client
    if _client is None:
        logger.info(
            f"[chroma] Initializing HTTP client: "
            f"{_settings.chroma_host}:{_settings.chroma_port}"
        )
        # Docker ChromaDB uses plain HTTP (no TLS) — ssl=True would fail the connection
        _client = chromadb.HttpClient(
            host=_settings.chroma_host,
            port=_settings.chroma_port,
            ssl=False,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("[chroma] HTTP client ready")
    return _client


def get_chroma_collection():
    """Return or create the singleton collection, reconnecting if the server restarted."""
    global _collection, _client
    if _collection is not None:
        try:
            _collection.count()
            return _collection
        except Exception:
            logger.warning("[chroma] Collection ping failed — reconnecting")
            _collection = None
            _client = None

    client = get_chroma_client()
    _collection = client.get_or_create_collection(
        name=_settings.chroma_collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Financial bulletin chunks — Ziraat & Halk Yatirim",
        },
    )
    count = _collection.count()
    logger.info(
        f"[chroma] Collection '{_settings.chroma_collection_name}' ready — "
        f"{count} documents"
    )
    return _collection


def get_collection_stats() -> dict:
    """Return basic stats for the active collection."""
    collection = get_chroma_collection()
    count = collection.count()
    return {
        "collection": _settings.chroma_collection_name,
        "chroma_host": _settings.chroma_host,
        "chroma_port": _settings.chroma_port,
        "document_count": count,
    }


def reset_collection() -> None:
    """Delete and recreate the collection — use with caution."""
    global _collection
    client = get_chroma_client()
    client.delete_collection(_settings.chroma_collection_name)
    _collection = None
    logger.warning(f"[chroma] Collection '{_settings.chroma_collection_name}' was reset")
    get_chroma_collection()
