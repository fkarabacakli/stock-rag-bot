from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from loguru import logger

from app.api.schemas import (
    FreeQueryRequest,
    HealthResponse,
    IngestTriggerResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    StatsResponse,
    WeeklyQueryRequest,
)
from app.config import get_settings

router = APIRouter()
_settings = get_settings()


# ── Sağlık & İstatistik ────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Ollama ve ChromaDB HTTP bağlantısını kontrol et."""
    from app.llm.client import health_check as ollama_health
    from app.vectorstore.client import get_collection_stats

    ollama_ok = await ollama_health()
    stats = get_collection_stats()

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        ollama_model=_settings.ollama_model,
        chroma_document_count=stats["document_count"],
        chroma_collection=stats["collection"],
        chroma_host=stats["chroma_host"],
        chroma_port=stats["chroma_port"],
    )


@router.get("/stats", response_model=StatsResponse, tags=["System"])
async def collection_stats():
    """ChromaDB koleksiyon istatistiklerini döndür."""
    from app.vectorstore.client import get_collection_stats
    return StatsResponse(**get_collection_stats())


# ── RAG Sorgu Uç Noktaları ─────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_analysis(req: QueryRequest):
    """
    RAG tabanlı hisse analizi sorgusu.
    Hisse kodu ve tarih aralığına göre filtreler, Türkçe yapılandırılmış yanıt üretir.
    """
    from app.rag.chain import query_analysis as run_query

    try:
        result = await run_query(
            query=req.query,
            stock_code=req.stock_code,
            source=req.source,
            days_back=req.days_back,
            model=req.model,
        )
    except Exception as exc:
        logger.error(f"[routes] /query başarısız: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG zinciri hatası: {str(exc)}",
        )

    return QueryResponse(
        success=result.raw_json.get("yeterli_veri", True),
        query=result.query,
        result=result.raw_json,
        sources=[SourceInfo(**s) for s in result.sources],
        model_used=result.model_used,
        chunks_retrieved=result.chunks_retrieved,
    )


@router.post("/query/weekly", response_model=QueryResponse, tags=["RAG"])
async def query_weekly(req: WeeklyQueryRequest):
    """Belirli bir BIST hissesi için haftalık çoklu kaynak özeti."""
    from app.rag.chain import query_weekly as run_weekly

    try:
        result = await run_weekly(
            stock_code=req.stock_code.upper(),
            model=req.model,
        )
    except Exception as exc:
        logger.error(f"[routes] /query/weekly başarısız: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Haftalık RAG hatası: {str(exc)}",
        )

    return QueryResponse(
        success=result.raw_json.get("yeterli_veri", True),
        query=result.query,
        result=result.raw_json,
        sources=[SourceInfo(**s) for s in result.sources],
        model_used=result.model_used,
        chunks_retrieved=result.chunks_retrieved,
    )


@router.post("/query/free", response_model=QueryResponse, tags=["RAG"])
async def free_query(req: FreeQueryRequest):
    """Hisse kodu filtresi olmadan serbest RAG sorgusu."""
    from app.rag.chain import free_query as run_free

    try:
        result = await run_free(
            query=req.query,
            days_back=req.days_back,
            model=req.model,
        )
    except Exception as exc:
        logger.error(f"[routes] /query/free başarısız: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Serbest sorgu hatası: {str(exc)}",
        )

    return QueryResponse(
        success=result.raw_json.get("yeterli_veri", True),
        query=result.query,
        result=result.raw_json,
        sources=[SourceInfo(**s) for s in result.sources],
        model_used=result.model_used,
        chunks_retrieved=result.chunks_retrieved,
    )


# ── Ingestion ──────────────────────────────────────────────────────────────────

@router.post("/ingest/trigger", response_model=IngestTriggerResponse, tags=["Ingestion"])
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    Ingestion pipeline'ı arka planda başlat.
    Hemen döner — ilerleme için /api/v1/stats kullanın.
    """
    from app.ingestion.pipeline import run_ingestion_pipeline

    async def _run():
        result = await run_ingestion_pipeline()
        logger.info(
            f"[routes] Manuel ingestion tamamlandı — "
            f"docs={result.total_documents}, upserted={result.upserted}"
        )

    background_tasks.add_task(_run)

    return IngestTriggerResponse(
        success=True,
        message="Ingestion pipeline arka planda başlatıldı. İlerleme için /api/v1/stats kontrol edin.",
    )


@router.post("/ingest/trigger/sync", response_model=IngestTriggerResponse, tags=["Ingestion"])
async def trigger_ingestion_sync():
    """
    Ingestion pipeline'ı senkron olarak çalıştır ve tamamlanmasını bekle.
    Test için uygundur; production'da async endpoint tercih edilir.
    """
    from app.ingestion.pipeline import run_ingestion_pipeline

    try:
        result = await run_ingestion_pipeline()
        return IngestTriggerResponse(
            success=len(result.errors) == 0,
            message="Ingestion tamamlandı",
            total_documents=result.total_documents,
            total_chunks=result.total_chunks,
            upserted=result.upserted,
            errors=result.errors,
        )
    except Exception as exc:
        logger.error(f"[routes] Senkron ingestion başarısız: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
