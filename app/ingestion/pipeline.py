"""
Ingestion pipeline orchestrator.

Flow: scrape → chunk → enrich tables → embed → upsert to ChromaDB
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.scrapers.base import BulletinDocument
from app.ingestion.scrapers.ziraat_yatirim import ZiraatYatirimScraper
from app.ingestion.table_parser import parse_and_format_tables

# Active scrapers — add new brokerages here
ACTIVE_SCRAPERS = [
    ZiraatYatirimScraper,
]


@dataclass
class IngestionResult:
    total_documents: int = 0
    total_chunks: int = 0
    upserted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


async def _scrape_all() -> list[BulletinDocument]:
    """Run all scrapers concurrently and collect results."""
    tasks = [scraper_cls().safe_fetch_bulletins() for scraper_cls in ACTIVE_SCRAPERS]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    all_docs: list[BulletinDocument] = []
    for doc_list in results:
        all_docs.extend(doc_list)
    logger.info(f"[pipeline] Total raw documents scraped: {len(all_docs)}")
    return all_docs


def _enrich_chunks_with_tables(chunks: list[Chunk]) -> list[Chunk]:
    """
    For each chunk, parse tables from its parent document's raw_html,
    append the human-readable table text, and store JSON in metadata.
    """
    enriched: list[Chunk] = []
    for chunk in chunks:
        try:
            table_text, table_json = parse_and_format_tables(
                chunk.metadata.get("raw_html", "")
            )
            if table_text:
                chunk.text = chunk.text + "\n\n" + table_text
                chunk.metadata["tables_json"] = table_json
        except Exception as exc:
            logger.debug(f"[pipeline] Table enrichment skipped for chunk {chunk.doc_id}: {exc}")
        enriched.append(chunk)
    return enriched


async def _embed_and_upsert(chunks: list[Chunk], result: IngestionResult) -> None:
    """Embed chunks and upsert to ChromaDB, writing counts/errors into `result`."""
    from app.vectorstore.client import get_chroma_collection
    from app.vectorstore.embeddings import get_embedding_function

    collection = get_chroma_collection()
    embed_fn = get_embedding_function()
    loop = asyncio.get_running_loop()

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            ids = [c.chromadb_id() for c in batch]
            texts = [c.embedding_text for c in batch]
            clean_metadatas = [{k: str(v) for k, v in c.metadata.items()} for c in batch]
            embeddings = await loop.run_in_executor(None, embed_fn, texts)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=clean_metadatas,
            )
            result.upserted += len(batch)
            logger.debug(f"[pipeline] Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")
        except Exception as exc:
            raw_error = str(exc)
            if "Collection expecting embedding with dimension" in raw_error and "got" in raw_error:
                error_msg = (
                    f"Batch {i // batch_size + 1} failed: {raw_error}. "
                    "Embedding dimension mismatch — reset the collection or change CHROMA_COLLECTION_NAME."
                )
            else:
                error_msg = f"Batch {i // batch_size + 1} failed: {raw_error}"
            logger.error(f"[pipeline] {error_msg}")
            result.errors.append(error_msg)
            result.skipped += len(batch)


async def run_ingestion_pipeline() -> IngestionResult:
    """
    Today's ingestion pipeline (all active scrapers, current bulletin only).
    Called by the daily scheduler or the /ingest/trigger endpoint.
    """
    result = IngestionResult()

    docs = await _scrape_all()
    result.total_documents = len(docs)
    if not docs:
        logger.warning("[pipeline] No documents scraped — aborting pipeline")
        return result

    chunks = chunk_documents(docs)
    logger.info(f"[pipeline] Generated {len(chunks)} chunks from {len(docs)} documents")
    chunks = _enrich_chunks_with_tables(chunks)
    result.total_chunks = len(chunks)
    if not chunks:
        logger.warning("[pipeline] No chunks generated — aborting pipeline")
        return result

    await _embed_and_upsert(chunks, result)
    logger.info(
        f"[pipeline] Done — upserted={result.upserted}, "
        f"skipped={result.skipped}, errors={len(result.errors)}"
    )
    return result


async def run_historical_ingestion_pipeline(days: int = 7) -> IngestionResult:
    """
    Fetch the last `days` Sabah Stratejisi bulletins and ingest them.

    Uses archive links (dropdown URLs / PDF hrefs) found on the main page.
    Other bulletin types (Teknik, Portföy) are image-based and are skipped here
    since no historical HTML is available for them.
    """
    result = IngestionResult()

    scraper = ZiraatYatirimScraper()
    try:
        async with scraper:
            docs = await scraper.fetch_historical_sabah_bulletins(days=days)
    except Exception as exc:
        error_msg = f"Historical scrape failed: {exc}"
        logger.error(f"[pipeline] {error_msg}", exc_info=True)
        result.errors.append(error_msg)
        return result

    result.total_documents = len(docs)
    if not docs:
        logger.warning(f"[pipeline] Historical: no documents scraped for last {days} days")
        return result

    chunks = chunk_documents(docs)
    logger.info(
        f"[pipeline] Historical: {len(chunks)} chunks from {len(docs)} documents "
        f"({days} days)"
    )
    chunks = _enrich_chunks_with_tables(chunks)
    result.total_chunks = len(chunks)
    if not chunks:
        return result

    await _embed_and_upsert(chunks, result)
    logger.info(
        f"[pipeline] Historical done — upserted={result.upserted}, "
        f"skipped={result.skipped}, errors={len(result.errors)}"
    )
    return result
