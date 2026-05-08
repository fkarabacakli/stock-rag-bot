"""
Ingestion pipeline orchestrator.

Flow: scrape → chunk → enrich tables → embed → upsert to ChromaDB
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.scrapers.base import BulletinDocument
from app.ingestion.scrapers.ziraat_yatirim import ZiraatYatirimScraper
from app.ingestion.table_parser import parse_and_format_tables

# Aktif scraper'lar — yeni kurum eklendikçe buraya ekleyin
ACTIVE_SCRAPERS = [
    ZiraatYatirimScraper,
]


@dataclass
class IngestionResult:
    total_documents: int = 0
    total_chunks: int = 0
    upserted: int = 0
    skipped: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


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


async def run_ingestion_pipeline() -> IngestionResult:
    """
    Full ingestion pipeline. Can be called by the scheduler or triggered
    via the API.
    """
    from app.vectorstore.client import get_chroma_collection
    from app.vectorstore.embeddings import get_embedding_function

    result = IngestionResult()

    # Step 1: Scrape
    docs = await _scrape_all()
    result.total_documents = len(docs)

    if not docs:
        logger.warning("[pipeline] No documents scraped — aborting pipeline")
        return result

    # Step 2: Chunk
    chunks = chunk_documents(docs)
    logger.info(f"[pipeline] Generated {len(chunks)} chunks from {len(docs)} documents")

    # Step 3: Enrich with table data
    chunks = _enrich_chunks_with_tables(chunks)
    result.total_chunks = len(chunks)

    if not chunks:
        logger.warning("[pipeline] No chunks generated — aborting pipeline")
        return result

    # Step 4: Embed and upsert to ChromaDB
    collection = get_chroma_collection()
    embed_fn = get_embedding_function()

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            ids = [c.chromadb_id() for c in batch]
            texts = [c.embedding_text for c in batch]
            metadatas = [c.metadata for c in batch]

            # Ensure metadata values are all strings (ChromaDB requirement)
            clean_metadatas = [
                {k: str(v) for k, v in m.items()} for m in metadatas
            ]

            embeddings = embed_fn(texts)

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
            if (
                "Collection expecting embedding with dimension" in raw_error
                and "got" in raw_error
            ):
                error_msg = (
                    f"Batch {i // batch_size + 1} failed: {raw_error}. "
                    "Embedding model dimension does not match existing Chroma collection. "
                    "Use a new CHROMA_COLLECTION_NAME or reset the current collection."
                )
            else:
                error_msg = f"Batch {i // batch_size + 1} failed: {raw_error}"
            logger.error(f"[pipeline] {error_msg}")
            result.errors.append(error_msg)
            result.skipped += len(batch)

    logger.info(
        f"[pipeline] Done — upserted={result.upserted}, "
        f"skipped={result.skipped}, errors={len(result.errors)}"
    )
    return result
