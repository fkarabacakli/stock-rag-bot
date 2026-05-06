"""
RAG zinciri: retrieve → context augment → LangChain ChatOllama → JSON parse.

Hibrit mimari:
  - Retrieval: özel MMR tabanlı retriever (SentenceTransformers + ChromaDB HTTP)
  - Generation: LangChain ChatOllama (localhost:11434, qwen2.5:7b, RTX 5070)

İki ana giriş noktası:
  - query_analysis()  → tek hisse analizi
  - query_weekly()    → haftalık çoklu kaynak özeti
  - free_query()      → serbest RAG sorgusu
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from app.config import get_settings
from app.llm.client import get_llm, parse_json_response
from app.llm.prompts import (
    NO_DATA_RESPONSE,
    SYSTEM_PROMPT,
    WEEKLY_SUMMARY_PROMPT,
    build_user_message,
    build_weekly_user_message,
)
from app.vectorstore.retriever import RetrievedChunk, retrieve

_SABAH_OVERVIEW_NEEDLES = (
    "hangi şirket",
    "hangi hisse",
    "bugün haber",
    "bugünkü sabah",
    "sabah stratejisi",
    "sabah rapor",
    "şirket haber",
    "hangi gelişme",
    "neyi var",
    "neler var",
    "raporda hangi",
    "bültende ne",
    "sabah bülten",
    "günlük bültende",
    "hangi kurum",
    "bülten özet",
)


def _sabah_overview_intent(query: str) -> bool:
    t = " ".join(query.lower().split())
    return any(n in t for n in _SABAH_OVERVIEW_NEEDLES)


def _apply_min_score(chunks: list[RetrievedChunk], min_score: float) -> list[RetrievedChunk]:
    if not chunks:
        return []
    passed = [c for c in chunks if c.score >= min_score]
    if passed:
        return passed
    best = max(c.score for c in chunks)
    logger.warning(
        f"[rag] Skor eşiği {min_score} üstü yok (en iyi {best:.3f}) — "
        f"{len(chunks)} chunk yine de kullanılıyor"
    )
    return sorted(chunks, key=lambda c: c.score, reverse=True)


@dataclass
class RAGResponse:
    raw_json: dict
    sources: list[dict]
    query: str
    model_used: str
    chunks_retrieved: int


def _format_chunk_for_context(chunk: RetrievedChunk) -> str:
    """Chunk'ı LLM context metnine dönüştür."""
    meta = chunk.metadata
    header_parts = [
        f"Source: {meta.get('source', 'unknown')}",
        f"Stock/ticker: {meta.get('stock_code', '') or meta.get('hisse', '') or '-'}",
        f"Company: {meta.get('sirket', '') or '-'}",
        f"Date: {meta.get('date', '?')}",
        f"Category: {meta.get('category', '?')}",
    ]
    if meta.get("bulten_turu"):
        header_parts.append(f"Bulletin: {meta['bulten_turu']}")
    if meta.get("section"):
        header_parts.append(f"Section: {meta['section']}")
    if meta.get("analyst"):
        header_parts.append(f"Analyst: {meta['analyst']}")

    return " | ".join(header_parts) + "\n" + chunk.text


def _chunks_to_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Chunk'lardan kaynak atıf listesi oluştur."""
    seen: set[tuple] = set()
    sources: list[dict] = []
    for c in chunks:
        key = (c.metadata.get("source"), c.metadata.get("date"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": c.metadata.get("source", "unknown"),
                "date": c.metadata.get("date", ""),
                "url": c.metadata.get("url", ""),
                "stock_code": c.metadata.get("stock_code", ""),
            })
    return sources


async def query_analysis(
    query: str,
    stock_code: Optional[str] = None,
    source: Optional[str] = None,
    days_back: int = 7,
    model: Optional[str] = None,
    sabah_overview: bool = False,
) -> RAGResponse:
    """
    RAG tabanlı hisse analizi sorgusu.

    Akış:
      1. ChromaDB'den alakalı chunk'ları getir (MMR re-rank)
      2. ChatOllama + sistem/kullanıcı mesajları → JSON
      3. JSON çıktıyı ayrıştır

    Args:
        query:      Doğal dil sorusu
        stock_code: Filtre için BIST kodu (ör. "THYAO")
        source:     Filtre için kaynak kurum (ör. "ziraat_yatirim")
        days_back:  Kaç gün geriye bakılsın
        model:      Ollama model override

    Returns:
        RAGResponse — JSON sonuç, kaynaklar ve meta veriler
    """
    settings = get_settings()
    llm = get_llm(model)

    retrieve_query = query
    if sabah_overview:
        retrieve_query = (
            f"{query} Sabah Stratejisi Ziraat Yatırım şirket haberleri "
            f"sektör haberleri günlük bülten"
        )

    top_k = settings.rag_sabah_overview_top_k if sabah_overview else None
    mmr_lambda = settings.rag_sabah_overview_mmr_lambda if sabah_overview else None

    chunks = retrieve(
        query=retrieve_query,
        stock_code=stock_code,
        source=source,
        days_back=days_back,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
    )
    chunks = _apply_min_score(chunks, settings.rag_min_chunk_score)

    logger.info(
        f"[rag] query='{query[:60]}' | stock={stock_code} | "
        f"sabah_overview={sabah_overview} | chunks={len(chunks)}"
    )

    if not chunks:
        logger.warning("[rag] İlgili chunk bulunamadı — NO_DATA_RESPONSE döndürülüyor")
        return RAGResponse(
            raw_json=json.loads(NO_DATA_RESPONSE),
            sources=[],
            query=query,
            model_used=llm.model,
            chunks_retrieved=0,
        )

    context_texts = [_format_chunk_for_context(c) for c in chunks]
    user_message = build_user_message(query, context_texts)

    logger.debug(f"[rag] LLM çağrılıyor → model={llm.model}")
    msg_out = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    raw_text = msg_out.content if isinstance(msg_out, AIMessage) else str(msg_out)
    parsed = parse_json_response(raw_text)

    if not parsed.get("hisse_kodu") and stock_code:
        parsed["hisse_kodu"] = stock_code.upper()

    return RAGResponse(
        raw_json=parsed,
        sources=_chunks_to_sources(chunks),
        query=query,
        model_used=llm.model,
        chunks_retrieved=len(chunks),
    )


async def query_weekly(
    stock_code: str,
    model: Optional[str] = None,
) -> RAGResponse:
    """
    Belirli bir hisse için haftalık çoklu kaynak özeti.

    Tüm kurumların son 7 günlük raporlarını sentezler.
    """
    from app.config import get_settings
    settings = get_settings()

    llm = get_llm(model)

    query = f"Bu hafta {stock_code} hissesine ait tüm analizler ve öneriler"

    chunks = retrieve(
        query=query,
        stock_code=stock_code,
        days_back=7,
        top_k=settings.retriever_top_k * 2,
    )
    chunks = _apply_min_score(chunks, settings.rag_min_chunk_score)

    logger.info(f"[rag] weekly | stock={stock_code} | chunks={len(chunks)}")

    if not chunks:
        return RAGResponse(
            raw_json=json.loads(NO_DATA_RESPONSE),
            sources=[],
            query=query,
            model_used=llm.model,
            chunks_retrieved=0,
        )

    context_texts = [_format_chunk_for_context(c) for c in chunks]
    user_message = build_weekly_user_message(query, context_texts, stock_code)

    msg_out = await llm.ainvoke(
        [SystemMessage(content=WEEKLY_SUMMARY_PROMPT), HumanMessage(content=user_message)]
    )
    raw_text = msg_out.content if isinstance(msg_out, AIMessage) else str(msg_out)
    parsed = parse_json_response(raw_text)

    if not parsed.get("hisse_kodu"):
        parsed["hisse_kodu"] = stock_code.upper()
    if not parsed.get("analiz_sayisi"):
        parsed["analiz_sayisi"] = len(chunks)

    return RAGResponse(
        raw_json=parsed,
        sources=_chunks_to_sources(chunks),
        query=query,
        model_used=llm.model,
        chunks_retrieved=len(chunks),
    )


async def free_query(
    query: str,
    days_back: int = 14,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> RAGResponse:
    """
    Hisse kodu filtresi olmadan serbest RAG sorgusu.
    Telegram serbest metin mesajları için kullanılır.
    """
    overview = _sabah_overview_intent(query)
    effective_days = max(days_back, 7) if overview else days_back
    return await query_analysis(
        query=query,
        stock_code=None,
        source=source,
        days_back=effective_days,
        model=model,
        sabah_overview=overview,
    )
