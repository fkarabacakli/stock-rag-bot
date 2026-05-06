import sys
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from loguru import logger

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level,
)

_scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Stock RAG Bot başlatılıyor...")
    logger.info(
        f"ChromaDB: {settings.chroma_host}:{settings.chroma_port} | "
        f"Ollama: {settings.ollama_base_url} | "
        f"Model: {settings.ollama_model}"
    )

    from scheduler.jobs import run_daily_ingestion

    _scheduler.add_job(
        run_daily_ingestion,
        trigger=CronTrigger(
            hour=settings.ingest_cron_hour,
            minute=settings.ingest_cron_minute,
            day_of_week="mon-fri",
        ),
        id="daily_ingestion",
        name="Günlük bülten ingestion",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info(
        f"Zamanlayıcı başlatıldı — ingestion saati: "
        f"{settings.ingest_cron_hour:02d}:{settings.ingest_cron_minute:02d} (Pzt-Cum)"
    )

    from app.bot.handlers import build_application

    tg_app = build_application()
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling başlatıldı")

    yield

    logger.info("Kapatılıyor...")
    await tg_app.updater.stop()
    await tg_app.stop()
    await tg_app.shutdown()
    _scheduler.shutdown(wait=False)
    logger.info("Kapatma tamamlandı")


app = FastAPI(
    title="Stock RAG Bot API",
    description="Yerel LLM destekli finansal bülten RAG sistemi — Ziraat & Halk Yatırım",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    return {
        "status": "ok",
        "service": "stock-rag-bot",
        "version": "0.1.0",
        "chroma": f"{settings.chroma_host}:{settings.chroma_port}",
        "ollama": settings.ollama_base_url,
        "model": settings.ollama_model,
    }
