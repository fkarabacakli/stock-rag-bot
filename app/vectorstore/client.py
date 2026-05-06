"""
ChromaDB HTTP client — Docker container'a bağlanır.

ChromaDB, localhost:8001 portunda HTTP server olarak çalışmaktadır.
Veri dizini: /home/furkan/Masaüstü/stock-rag-bot/chroma_data (volume mount)
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.config import get_settings

_settings = get_settings()
_client = None  # chromadb ClientAPI (HttpClient döner)
_collection = None


def get_chroma_client():
    """Docker ChromaDB HTTP server'a singleton bağlantı döndür."""
    global _client
    if _client is None:
        logger.info(
            f"[chroma] HTTP client başlatılıyor: "
            f"{_settings.chroma_host}:{_settings.chroma_port}"
        )
        # Docker ChromaDB düz HTTP kullanır (TLS yok) — ssl=True olursa bağlantı başarısız olur
        _client = chromadb.HttpClient(
            host=_settings.chroma_host,
            port=_settings.chroma_port,
            ssl=False,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("[chroma] HTTP client hazır")
    return _client


def get_chroma_collection():
    """Singleton koleksiyonu döndür veya oluştur."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=_settings.chroma_collection_name,
            metadata={
                "hnsw:space": "cosine",
                "description": "Finansal bülten chunk'ları — Ziraat & Halk Yatırım",
            },
        )
        count = _collection.count()
        logger.info(
            f"[chroma] Koleksiyon '{_settings.chroma_collection_name}' hazır — "
            f"{count} döküman"
        )
    return _collection


def get_collection_stats() -> dict:
    """Aktif koleksiyonun temel istatistiklerini döndür."""
    collection = get_chroma_collection()
    count = collection.count()
    return {
        "collection": _settings.chroma_collection_name,
        "chroma_host": _settings.chroma_host,
        "chroma_port": _settings.chroma_port,
        "document_count": count,
    }


def reset_collection() -> None:
    """Koleksiyonu sil ve yeniden oluştur — dikkatli kullan."""
    global _collection
    client = get_chroma_client()
    client.delete_collection(_settings.chroma_collection_name)
    _collection = None
    logger.warning(f"[chroma] Koleksiyon '{_settings.chroma_collection_name}' sıfırlandı")
    get_chroma_collection()
