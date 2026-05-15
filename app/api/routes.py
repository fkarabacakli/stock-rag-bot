from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from loguru import logger

from app.api.schemas import (
    CollectionResetResponse,
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


# ── Health & Statistics ───────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check Ollama and ChromaDB HTTP connectivity."""
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
    """Return ChromaDB collection statistics."""
    from app.vectorstore.client import get_collection_stats
    return StatsResponse(**get_collection_stats())


# ── RAG Query Endpoints ───────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_analysis(req: QueryRequest):
    """
    RAG-based stock analysis query.
    Filters by stock code and date range, and returns a structured Turkish response.
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
        logger.error(f"[routes] /query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG chain error: {str(exc)}",
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
    """Weekly multi-source summary for a specific BIST stock."""
    from app.rag.chain import query_weekly as run_weekly

    try:
        result = await run_weekly(
            stock_code=req.stock_code.upper(),
            model=req.model,
        )
    except Exception as exc:
        logger.error(f"[routes] /query/weekly failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weekly RAG error: {str(exc)}",
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
    """Free-form RAG query without stock-code filtering."""
    from app.rag.chain import free_query as run_free

    try:
        result = await run_free(
            query=req.query,
            days_back=req.days_back,
            model=req.model,
            source=req.source,
        )
    except Exception as exc:
        logger.error(f"[routes] /query/free failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Free query error: {str(exc)}",
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
    Start the ingestion pipeline in the background.
    Returns immediately — use /api/v1/stats to track progress.
    """
    from app.ingestion.pipeline import run_ingestion_pipeline

    async def _run():
        result = await run_ingestion_pipeline()
        logger.info(
            f"[routes] Manual ingestion completed — "
            f"docs={result.total_documents}, upserted={result.upserted}"
        )

    background_tasks.add_task(_run)

    return IngestTriggerResponse(
        success=True,
        message="Ingestion pipeline started in background. Check /api/v1/stats for progress.",
    )


@router.post("/collection/reset", response_model=CollectionResetResponse, tags=["System"])
async def reset_collection():
    """
    Wipe the ChromaDB collection entirely.
    All stored bulletins are deleted. Run ingestion afterwards to repopulate.
    """
    from app.vectorstore.client import get_collection_stats
    from app.vectorstore.client import reset_collection as _reset

    prev = get_collection_stats()["document_count"]
    _reset()
    return CollectionResetResponse(
        success=True,
        message="Collection wiped. Run /ingest/trigger to repopulate.",
        previous_count=prev,
    )


@router.post("/ingest/trigger/sync", response_model=IngestTriggerResponse, tags=["Ingestion"])
async def trigger_ingestion_sync():
    """
    Run the ingestion pipeline synchronously and wait for completion.
    Useful for testing; async endpoint is preferred in production.
    """
    from app.ingestion.pipeline import run_ingestion_pipeline

    try:
        result = await run_ingestion_pipeline()
        return IngestTriggerResponse(
            success=len(result.errors) == 0,
            message="Ingestion completed",
            total_documents=result.total_documents,
            total_chunks=result.total_chunks,
            upserted=result.upserted,
            errors=result.errors,
        )
    except Exception as exc:
        logger.error(f"[routes] Synchronous ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
