"""
RAG chain: retrieve → context augment → LangChain ChatOllama → JSON parse.

Hybrid architecture:
  - Retrieval: custom MMR-based retriever (SentenceTransformers + ChromaDB HTTP)
  - Generation: LangChain ChatOllama (localhost:11434, qwen2.5:7b, RTX 5070)

Main entry points:
  - query_analysis()  → single-stock analysis
  - query_weekly()    → weekly multi-source summary
  - free_query()      → free-form RAG query
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
        f"{len(chunks)} chunk yine de kullaniliyor"
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
    """Convert a chunk into LLM context text."""
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
    """Build a source citation list from chunks."""
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


def _apply_sabah_overview_fallback(parsed: dict, chunks: list[RetrievedChunk]) -> dict:
    """
    For morning overview questions, sometimes model returns valid JSON but leaves
    summary/list fields empty. Build a lightweight fallback from retrieved chunks.
    """
    if parsed.get("ozet") or parsed.get("sirket_haber_ozetleri"):
        return parsed

    items: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        meta = chunk.metadata
        ticker = (meta.get("stock_code") or meta.get("hisse") or "").strip().upper()
        company = (meta.get("sirket") or meta.get("company") or "").strip()
        key = (ticker, company.lower())
        if key in seen:
            continue
        seen.add(key)

        short_text = " ".join((chunk.text or "").strip().split())
        if short_text:
            short_text = short_text[:220].rstrip(" .,:;") + "."
        else:
            short_text = "Bu kayit ilgili bültende öne çikan başliklar arasinda yer aliyor."

        items.append(
            {
                "hisse_kodu": ticker or None,
                "sirket_adi": company or ("Belirtilmemiş" if not ticker else ""),
                "kisa_ozet": short_text,
            }
        )
        if len(items) >= 18:
            break

    if items:
        first_labels = []
        for it in items[:5]:
            label = it["hisse_kodu"] or it["sirket_adi"] or "Kayit"
            first_labels.append(str(label))
        parsed["ozet"] = (
            f"Sabah bültenlerinde öne çikan {len(items)} şirket/kurum bulundu: "
            + ", ".join(first_labels)
            + "."
        )
        parsed["sirket_haber_ozetleri"] = items

    if not parsed.get("kaynaklar"):
        parsed["kaynaklar"] = [
            f"{s.get('source', 'unknown')} - {s.get('date', '')}".strip(" -")
            for s in _chunks_to_sources(chunks)[:4]
        ]

    return parsed


async def query_analysis(
    query: str,
    stock_code: Optional[str] = None,
    source: Optional[str] = None,
    days_back: int = 7,
    model: Optional[str] = None,
    sabah_overview: bool = False,
) -> RAGResponse:
    """
    RAG-based stock analysis query.

    Flow:
      1. Retrieve relevant chunks from ChromaDB (MMR re-rank)
      2. ChatOllama + system/user messages → JSON
      3. Parse the JSON output

    Args:
        query:      Natural language query
        stock_code: Optional BIST ticker filter (e.g. "THYAO")
        source:     Optional brokerage filter (e.g. "ziraat_yatirim")
        days_back:  How many days back to search
        model:      Ollama model override

    Returns:
        RAGResponse — JSON result, sources, and metadata
    """
    settings = get_settings()
    llm = get_llm(model)

    retrieve_query = query
    if sabah_overview:
        retrieve_query = (
            f"{query} Sabah Stratejisi Ziraat Yatirim şirket haberleri "
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
        logger.warning("[rag] No relevant chunks found — returning NO_DATA_RESPONSE")
        return RAGResponse(
            raw_json=json.loads(NO_DATA_RESPONSE),
            sources=[],
            query=query,
            model_used=llm.model,
            chunks_retrieved=0,
        )

    context_texts = [_format_chunk_for_context(c) for c in chunks]
    user_message = build_user_message(query, context_texts)

    logger.debug(f"[rag] Calling LLM -> model={llm.model}")
    msg_out = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    raw_text = msg_out.content if isinstance(msg_out, AIMessage) else str(msg_out)
    parsed = parse_json_response(raw_text)

    if sabah_overview:
        parsed = _apply_sabah_overview_fallback(parsed, chunks)

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
    Weekly multi-source summary for a specific stock.

    Synthesizes last 7-day reports across all brokerages.
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
    Free-form RAG query without a stock-code filter.
    Used for Telegram free-text messages.
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
