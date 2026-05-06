"""
Embedding function singleton.

Uses sentence-transformers with a multilingual model that handles both
Turkish bulletin content and mixed Turkish/English queries.

Model: paraphrase-multilingual-mpnet-base-v2
  - 768-dimensional embeddings
  - ~420 MB disk, ~1.5 GB RAM
  - Supports 50+ languages including Turkish
"""
from __future__ import annotations

from typing import Callable

from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import get_settings

_settings = get_settings()
_model: SentenceTransformer | None = None


def _resolve_torch_device() -> str:
    """embedding_device: auto | cuda | cpu"""
    want = (_settings.embedding_device or "auto").lower().strip()
    try:
        import torch
    except ImportError:
        return "cpu"

    if want == "cpu":
        return "cpu"
    if want == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "[embeddings] embedding_device=cuda ama torch.cuda.is_available()=False "
            "(PyTorch CPU paketi veya sürücü yok) — CPU kullanılıyor"
        )
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        device = _resolve_torch_device()
        logger.info(f"[embeddings] Loading model: {_settings.embedding_model} (device={device})")
        _model = SentenceTransformer(_settings.embedding_model, device=device)
        logger.info(
            f"[embeddings] Model loaded — dim={_model.get_sentence_embedding_dimension()}, device={device}"
        )
    return _model


def get_embedding_function() -> Callable[[list[str]], list[list[float]]]:
    """
    Return a callable that takes a list of strings and returns embeddings.
    Compatible with ChromaDB's upsert/query interface.
    """
    model = _load_model()

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
        return vectors.tolist()

    return embed


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _load_model()
    vector = model.encode(query, show_progress_bar=False)
    return vector.tolist()
