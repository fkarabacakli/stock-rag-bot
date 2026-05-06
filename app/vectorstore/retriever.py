"""
Retriever with metadata filtering and MMR (Maximum Marginal Relevance).

Supports:
  - Exact metadata filters (source, stock_code, date range)
  - MMR-style re-ranking for diversity (avoids redundant chunks)
  - Time-series queries (e.g., "last 7 days")
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from app.config import get_settings
from app.vectorstore.client import get_chroma_collection
from app.vectorstore.embeddings import embed_query

_settings = get_settings()


def _as_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    return list(raw)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float
    chunk_id: str


def _build_where_filter(source: Optional[str] = None) -> Optional[dict]:
    """
    ChromaDB metadata filtresi (yalnızca kaynak).

    Tarih Chroma where içinde kullanılmaz — Chroma 1.x ile uyumsuzluk / boş sonuç önlenir;
    tarih retrieve() içinde Python'da uygulanır.
    """
    if source:
        return {"source": {"$eq": source}}
    return None


def _meta_calendar_date(meta: dict) -> Optional[date]:
    """Chunk metadata'dan takvim günü (date / tarih ISO)."""
    raw = meta.get("date") or meta.get("tarih")
    if raw is None:
        return None
    s = str(raw).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _metadata_in_date_range(
    meta: dict,
    date_from: Optional[date],
    date_to: Optional[date],
) -> bool:
    d = _meta_calendar_date(meta)
    if d is None:
        return True
    if date_from is not None and d < date_from:
        return False
    if date_to is not None and d > date_to:
        return False
    return True


def _chunk_matches_stock(chunk: RetrievedChunk, stock_code: str) -> bool:
    """Match by exact stock_code or by mentioned_tickers metadata."""
    target = stock_code.upper()
    meta_stock = str(chunk.metadata.get("stock_code", "")).upper()
    if meta_stock == target:
        return True
    mentioned = str(chunk.metadata.get("mentioned_tickers", "")).upper()
    if not mentioned:
        return False
    mentioned_set = {t.strip() for t in mentioned.split(",") if t.strip()}
    return target in mentioned_set


def _mmr_rerank(
    query_vec: list[float],
    candidate_chunks: list[RetrievedChunk],
    candidate_embeddings: list[list[float]],
    top_k: int,
    lambda_param: float,
) -> list[RetrievedChunk]:
    """
    Maximum Marginal Relevance re-ranking.

    Balances relevance to query vs. diversity among selected chunks.
    lambda_param=1.0 → pure relevance; lambda_param=0.0 → pure diversity.
    """
    if not candidate_chunks:
        return []

    query_arr = np.array(query_vec, dtype=float)
    cand_arrs = np.array(candidate_embeddings, dtype=float)

    # Cosine similarity: query vs candidates
    query_norm = np.linalg.norm(query_arr)
    cand_norms = np.linalg.norm(cand_arrs, axis=1, keepdims=True)
    cand_arrs_norm = cand_arrs / (cand_norms + 1e-9)
    query_sims = cand_arrs_norm @ (query_arr / (query_norm + 1e-9))

    selected_indices: list[int] = []
    remaining = list(range(len(candidate_chunks)))

    for _ in range(min(top_k, len(candidate_chunks))):
        if not remaining:
            break

        if not selected_indices:
            # First: pick highest relevance
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            # MMR: relevance - lambda * max_similarity_to_selected
            selected_arrs = cand_arrs_norm[selected_indices]
            mmr_scores = []
            for i in remaining:
                rel = lambda_param * query_sims[i]
                red = (1 - lambda_param) * float(np.max(selected_arrs @ cand_arrs_norm[i]))
                mmr_scores.append((i, rel - red))
            best = max(mmr_scores, key=lambda x: x[1])[0]

        selected_indices.append(best)
        remaining.remove(best)

    return [candidate_chunks[i] for i in selected_indices]


def retrieve(
    query: str,
    stock_code: Optional[str] = None,
    source: Optional[str] = None,
    days_back: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    top_k: Optional[int] = None,
    use_mmr: bool = True,
    mmr_lambda: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query with optional metadata filters.

    Args:
        query:      Natural language query
        stock_code: Filter to a specific ticker (e.g., "THYAO")
        source:     Filter to a specific brokerage (e.g., "ziraat_yatirim")
        days_back:  Restrict to the last N days
        date_from:  Explicit start date filter
        date_to:    Explicit end date filter
        top_k:       Number of final results (default from config)
        use_mmr:     Apply MMR re-ranking for diversity
        mmr_lambda:  MMR çeşitliliği (None = config)

    Returns:
        List of RetrievedChunk objects sorted by relevance/MMR score.
    """
    k = top_k or _settings.retriever_top_k
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.warning("[retriever] Collection is empty — run ingestion first")
        return []

    # Resolve date range
    resolved_date_from = date_from
    if days_back and not date_from:
        resolved_date_from = date.today() - timedelta(days=days_back)

    where = _build_where_filter(source=source)

    # Embed query
    query_vec = embed_query(query)

    # Tarih filtresi Python'da — aday sayısını artır (filtre sonrası yeterli kalsın)
    date_trim = resolved_date_from is not None or date_to is not None
    mult = 8 if date_trim else 3
    n_candidates = min(max(k * mult, 48), collection.count())

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=n_candidates,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
    except Exception as exc:
        logger.error(f"[retriever] ChromaDB query failed: {exc}")
        return []

    docs = _as_list(results.get("documents", [[]])[0])
    metas = _as_list(results.get("metadatas", [[]])[0])
    distances = _as_list(results.get("distances", [[]])[0])
    embeddings = _as_list(results.get("embeddings", [[]])[0])
    ids = _as_list(results.get("ids", [[]])[0])

    if len(docs) == 0:
        logger.info(f"[retriever] No results for query: '{query[:60]}...'")
        return []

    # Convert cosine distance to similarity score
    candidates = [
        RetrievedChunk(
            text=doc,
            metadata=meta,
            score=1.0 - dist,
            chunk_id=chunk_id,
        )
        for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids)
    ]

    if date_trim:
        kept: list[RetrievedChunk] = []
        kept_emb: list = []
        for cand, emb in zip(candidates, embeddings):
            if _metadata_in_date_range(cand.metadata, resolved_date_from, date_to):
                kept.append(cand)
                kept_emb.append(emb)
        candidates = kept
        embeddings = kept_emb
        if not candidates:
            logger.info(
                f"[retriever] Tarih filtresi sonrası aday kalmadı "
                f"(from={resolved_date_from}, to={date_to})"
            )
            return []

    if stock_code and stock_code.upper() != "GENEL":
        filtered_pairs = [
            (cand, emb)
            for cand, emb in zip(candidates, embeddings)
            if _chunk_matches_stock(cand, stock_code)
        ]
        if filtered_pairs:
            candidates = [c for c, _ in filtered_pairs]
            embeddings = [e for _, e in filtered_pairs]
        else:
            candidates = []
            embeddings = []

    lambda_m = mmr_lambda if mmr_lambda is not None else _settings.retriever_mmr_lambda
    if use_mmr and len(embeddings) > 0 and candidates:
        final = _mmr_rerank(
            query_vec=query_vec,
            candidate_chunks=candidates,
            candidate_embeddings=embeddings,
            top_k=k,
            lambda_param=lambda_m,
        )
    else:
        final = candidates[:k]

    logger.debug(
        f"[retriever] Query='{query[:40]}' → {len(final)} chunks "
        f"(filter: stock={stock_code}, source={source}, days_back={days_back})"
    )
    return final
