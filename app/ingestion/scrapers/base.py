from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


@dataclass
class BulletinDocument:
    """Single bulletin document scraped from a brokerage website."""

    title: str
    stock_code: str
    category: str
    raw_html: str
    date: date
    source: str
    url: str
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        from app.ingestion.scrapers.semantic import KURUM_BY_SOURCE

        sc = self.stock_code.upper()
        return {
            "title": self.title,
            "stock_code": sc,
            "hisse": sc,
            "category": self.category,
            "kategori": self.category,
            "date": self.date.isoformat(),
            "tarih": self.date.isoformat(),
            "source": self.source,
            "kurum": KURUM_BY_SOURCE.get(self.source, self.source),
            "url": self.url,
            **self.extra_metadata,
        }


class BaseScraper(ABC):
    """Abstract base class for all brokerage bulletin scrapers."""

    SOURCE_NAME: str = "unknown"
    BASE_URL: str = ""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BaseScraper":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.scraper_request_timeout),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch(self, url: str, *, extra_headers: Optional[Mapping[str, str]] = None) -> str:
        """Fetch a URL with retry logic. Per-request headers merge with the client defaults."""
        assert self._client is not None, "Use as async context manager"
        logger.debug(f"[{self.SOURCE_NAME}] Fetching: {url}")
        response = await self._client.get(url, headers=dict(extra_headers) if extra_headers else None)
        response.raise_for_status()
        return response.text

    @abstractmethod
    async def fetch_bulletins(self) -> list[BulletinDocument]:
        """
        Fetch today's bulletin documents from the brokerage website.
        Returns a list of BulletinDocument instances.
        """
        ...

    async def safe_fetch_bulletins(self) -> list[BulletinDocument]:
        """Wraps fetch_bulletins with top-level error handling."""
        try:
            async with self:
                docs = await self.fetch_bulletins()
                logger.info(f"[{self.SOURCE_NAME}] Fetched {len(docs)} documents")
                return docs
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Failed to fetch bulletins: {exc}")
            return []
