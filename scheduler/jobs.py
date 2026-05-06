"""
APScheduler iş tanımları.

İşler app/main.py'deki lifespan context'inde kayıt edilir.
"""
from __future__ import annotations

from loguru import logger


async def run_daily_ingestion() -> None:
    """
    Günlük bülten ingestion işi.
    Her hafta içi sabah çalışır — scrape, chunk, embed ve ChromaDB'ye kaydet.
    """
    logger.info("[scheduler] Günlük ingestion işi başlatılıyor...")
    try:
        from app.ingestion.pipeline import run_ingestion_pipeline

        result = await run_ingestion_pipeline()
        logger.info(
            f"[scheduler] Günlük ingestion tamamlandı — "
            f"docs={result.total_documents}, "
            f"chunks={result.total_chunks}, "
            f"upserted={result.upserted}, "
            f"errors={len(result.errors)}"
        )
        if result.errors:
            for err in result.errors:
                logger.warning(f"[scheduler] Pipeline hatası: {err}")
    except Exception as exc:
        logger.error(f"[scheduler] Günlük ingestion başarısız: {exc}", exc_info=True)
