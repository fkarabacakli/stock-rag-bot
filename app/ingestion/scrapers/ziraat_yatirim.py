"""
Ziraat Yatirim bulletin scraper.

Primary: Sabah Stratejisi (daily morning strategy note)
  https://www.ziraatyatirim.com.tr/tr/sabah-stratejisi

Content lives under #ContentSection > .sub-page-content (not the old hisse-raporlari list layout).

Company reports require a separate page/selectors; to avoid noise, we only ingest
Sabah Stratejisi as a single BulletinDocument.
"""
from __future__ import annotations

import base64
import json
import re
import unicodedata
from datetime import date
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from app.config import get_settings
from app.ingestion.scrapers.base import BaseScraper, BulletinDocument
from app.ingestion.scrapers.semantic import (
    clean_ws,
    ticker_from_parenthetical,
    wrap_metin_paragraph,
)
from app.ingestion.scrapers.semantic import KURUM_BY_SOURCE

settings = get_settings()

# Title line on page: "Sabah Stratejisi - 24 / 04 / 2026" (nbsp may appear as space after BS parse)
_SABAH_DATE_RE = re.compile(
    r"Sabah\s+Stratejisi\s*[-–]\s*(\d{1,2})\s*[/\s]+\s*(\d{1,2})\s*[/\s]+\s*(\d{4})",
    re.IGNORECASE,
)
_GENERIC_DMY_RE = re.compile(r"(\d{1,2})\s*[/.\s]+\s*(\d{1,2})\s*[/.\s]+\s*(\d{4})")


def _parse_date_from_text(text: str) -> date | None:
    """Return the first DD/MM/YYYY or DD.MM.YYYY date found in plain text."""
    m = _GENERIC_DMY_RE.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None
_TICKER_RE = re.compile(r"\b[A-Z]{3,5}\b")
# "Arcelik (ARCLK, Notr): ..." — keep full parenthetical (rating) in metin
_COMPANY_ITEM_RE = re.compile(
    r"^\s*(?P<company>.+?)\s*\((?P<paren>[^)]+)\)\s*:\s*(?P<body>.+)$",
    re.DOTALL,
)
# "Ziraat Bankasi: ..." / "SPK: ..." — no ticker in parentheses
_COMPANY_COLON_NO_TICKER_RE = re.compile(
    r"^\s*(?P<company>"
    r"[A-ZÇĞIÖŞÜI][\w'.-]*(?:\s+[\w'.-]+)*"  # Is GYO, Ziraat Bankasi
    r"|[A-ZÇĞIÖŞÜI]{2,8}"  # short abbreviation: SPK, TCMB
    r")\s*:\s*(?P<body>\S.+)$",
    re.DOTALL,
)
# Do not treat sentence continuations like "Bu ceyrekte:" as company names
_BAD_STANDALONE_COMPANY_START = frozenset(
    {
        # All entries are pre-ascii-folded so comparison with _ascii_fold(word) works correctly.
        "bu",
        "buna",
        "bunu",
        "boylece",        # böylece
        "boylelikle",     # böylelikle
        "ayrica",         # ayrıca
        "diger",          # diğer
        "ote",            # öte
        "yani",
        "neticede",
        "sonucta",        # sonuçta
        "dolayisiyla",    # dolayısıyla
        "ozetle",         # özetle
        "nihayetinde",
        "bununla",
        "bundan",
        "cogu",           # çoğu
        "bazi",           # bazı
        "soyle",          # şöyle
        "ayri",           # ayrı
        "ilk",
        "ikinci",
        "ucuncu",         # üçüncü
        "gecen",          # geçen
        "gectigimiz",     # geçtiğimiz
    }
)


def _standalone_company_plausible(company: str) -> bool:
    parts = company.strip().split()
    if not parts or len(parts) > 10:
        return False
    if "(" in company or ")" in company:
        return False
    first = _ascii_fold(parts[0])
    if first in _BAD_STANDALONE_COMPANY_START:
        return False
    return len(company.strip()) >= 2


def _parse_sabah_date_from_html(soup: BeautifulSoup) -> date | None:
    """Parse bulletin date from the first Sabah Stratejisi heading."""
    for h3 in soup.find_all("h3"):
        t = h3.get_text(" ", strip=True)
        m = _SABAH_DATE_RE.search(t)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mo, d)
            except ValueError:
                continue
    return None


def _parse_any_date_from_html(soup: BeautifulSoup) -> date | None:
    """Parse first DD/MM/YYYY-like date seen in headings/text."""
    for tag in soup.find_all(["h1", "h2", "h3", "p", "span"]):
        t = tag.get_text(" ", strip=True)
        m = _GENERIC_DMY_RE.search(t)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mo, d)
            except ValueError:
                continue
    return None


def _extract_main_content_html(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Return (raw_html_fragment, page_title) for Sabah Stratejisi body.

    Prefers the first editorial column inside #ContentSection.
    """
    section = soup.select_one("#ContentSection")
    if not section:
        section = soup

    # Inner column that holds the h3 + paragraphs (not the document filter form)
    inner = section.select_one(".sub-page-content div.row > div.col-xs-12")
    if inner is None:
        inner = section.select_one(".sub-page-content") or section.select_one(
            ".sub-page-container .col-xs-12"
        )

    if inner is None:
        return "", ""

    # Title: first meaningful h3 text
    title = "Sabah Stratejisi"
    h3_first = inner.find("h3")
    if h3_first:
        title = " ".join(h3_first.get_text().split())[:200]

    # Strip trailing filter UI: remove siblings after documentFilter if present
    fragment_soup = BeautifulSoup(str(inner), "lxml")
    for noise in fragment_soup.select(".documentFilter, .form-page-wrapper"):
        noise.decompose()

    return str(fragment_soup), title


def _extract_mentioned_tickers(raw_html: str) -> list[str]:
    """Extract likely BIST tickers mentioned in the bulletin text."""
    text = BeautifulSoup(raw_html, "lxml").get_text(" ", strip=True)
    candidates = _TICKER_RE.findall(text.upper())
    # Keep unique order and filter obvious false positives.
    ignore = {"BIST", "TL", "USD", "EUR", "TRY", "ABD", "FED", "TCMB", "GENEL"}
    seen: set[str] = set()
    tickers: list[str] = []
    for c in candidates:
        if c in ignore:
            continue
        if c not in seen:
            seen.add(c)
            tickers.append(c)
    return tickers


def _ascii_fold(s: str) -> str:
    """Lowercase + strip combining marks so 'SIRKET' matches 'sirket'."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.casefold()


def _normalize_category(title: str) -> str:
    """Normalize section titles into stable metadata categories."""
    t = _ascii_fold(" ".join(title.split()))
    mapping = {
        "sirket haberleri": "Sirket_Haberleri",
        "sektor haberleri": "Sektor_Haberleri",
        "makro": "Makro",
        "emtia": "Emtia",
        "kur ve emtia": "Kur_ve_Emtia",
        "yurt disi piyasalar": "Yurt_Disi_Piyasalar",
    }
    for key, val in mapping.items():
        if key in t:
            return val
    slug = "_".join(_ascii_fold(title).split())
    return slug[:80] if slug else "Genel"


def _extract_section_records(raw_fragment: str) -> list[dict]:
    """
    Parse Sabah Stratejisi HTML into semantic records:
      [{"metin": "...", "metadata": {...}}, ...]

    Main target: keep company name + ticker + analysis text together.
    Ziraat splits long items across multiple <p> siblings; we merge those
    into a single record until the next "Sirket (TICKER, ...): ..." or
    "Kurum Adi: ..." (without ticker) line.
    """
    soup = BeautifulSoup(raw_fragment, "lxml")
    records: list[dict] = []

    def _flush(buf: dict | None) -> None:
        if buf and clean_ws(buf.get("metin", "")):
            records.append(buf)

    for h3 in soup.find_all("h3"):
        section_title = " ".join(h3.get_text(" ", strip=True).split())
        category = _normalize_category(section_title)

        current: dict | None = None
        general_buf: dict | None = None

        sibling = h3.find_next_sibling()
        while sibling and sibling.name != "h3":
            if sibling.name in {"p", "li"}:
                text = clean_ws(sibling.get_text(" ", strip=True))
                if len(text) < 20:
                    sibling = sibling.find_next_sibling()
                    continue

                m = _COMPANY_ITEM_RE.match(text)
                if m:
                    _flush(general_buf)
                    general_buf = None
                    _flush(current)

                    company = clean_ws(m.group("company"))
                    paren = clean_ws(m.group("paren"))
                    body = clean_ws(m.group("body"))
                    ticker = ticker_from_parenthetical(paren)
                    metin = f"{company} ({paren}): {body}"
                    current = {
                        "metin": metin,
                        "metadata": {
                            "kategori": category,
                            "hisse": ticker,
                            "sirket": company,
                        },
                    }
                elif (
                    (m2 := _COMPANY_COLON_NO_TICKER_RE.match(text))
                    and _standalone_company_plausible(m2.group("company"))
                ):
                    _flush(general_buf)
                    general_buf = None
                    _flush(current)

                    company = clean_ws(m2.group("company"))
                    body = clean_ws(m2.group("body"))
                    metin = f"{company}: {body}"
                    current = {
                        "metin": metin,
                        "metadata": {
                            "kategori": category,
                            "hisse": "",
                            "sirket": company,
                        },
                    }
                else:
                    if current is not None:
                        current["metin"] = clean_ws(current["metin"] + " " + text)
                    else:
                        if general_buf is None:
                            general_buf = {
                                "metin": text,
                                "metadata": {
                                    "kategori": category,
                                    "hisse": "GENEL",
                                    "sirket": "",
                                },
                            }
                        else:
                            general_buf["metin"] = clean_ws(general_buf["metin"] + " " + text)
            sibling = sibling.find_next_sibling()

        _flush(current)
        _flush(general_buf)

    return records


class ZiraatYatirimScraper(BaseScraper):
    SOURCE_NAME = "ziraat_yatirim"
    BASE_URL = "https://www.ziraatyatirim.com.tr"
    SABAH_STRATEJISI_URL = f"{BASE_URL}/tr/sabah-stratejisi"
    GUNLUK_TEKNIK_BULTEN_URL = f"{BASE_URL}/tr/gunluk-teknik-bulten"
    HISSE_ONERI_PORTFOYU_URL = f"{BASE_URL}/tr/hisse-oneri-portfoyu"
    HAFTALIK_TEKNIK_HISSE_URL = f"{BASE_URL}/tr/haftalik-teknik-hisse-onerileri"

    async def fetch_bulletins(self) -> list[BulletinDocument]:
        documents: list[BulletinDocument] = []

        try:
            html = await self._fetch(
                self.SABAH_STRATEJISI_URL,
                extra_headers={"Referer": f"{self.BASE_URL}/tr/arastirma"},
            )
            sabah_docs = self._parse_sabah_stratejisi(html, source_url=self.SABAH_STRATEJISI_URL)
            documents.extend(sabah_docs)
            logger.info(f"[{self.SOURCE_NAME}] Sabah Stratejisi: {len(sabah_docs)} document(s)")
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Sabah Stratejisi fetch failed: {exc}")

        try:
            html = await self._fetch(
                self.GUNLUK_TEKNIK_BULTEN_URL,
                extra_headers={"Referer": f"{self.BASE_URL}/tr/arastirma"},
            )
            teknik_docs = await self._parse_gunluk_teknik_bulten(html)
            documents.extend(teknik_docs)
            logger.info(f"[{self.SOURCE_NAME}] Günlük Teknik Bülten: {len(teknik_docs)} document(s)")
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Günlük Teknik Bülten fetch failed: {exc}")

        try:
            html = await self._fetch(
                self.HISSE_ONERI_PORTFOYU_URL,
                extra_headers={"Referer": f"{self.BASE_URL}/tr/arastirma"},
            )
            portfoy_docs = await self._parse_hisse_oneri_portfoyu(html)
            documents.extend(portfoy_docs)
            logger.info(f"[{self.SOURCE_NAME}] Hisse Öneri Portföyü: {len(portfoy_docs)} document(s)")
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Hisse Öneri Portföyü fetch failed: {exc}")

        try:
            html = await self._fetch(
                self.HAFTALIK_TEKNIK_HISSE_URL,
                extra_headers={"Referer": f"{self.BASE_URL}/tr/arastirma"},
            )
            haftalik_docs = await self._parse_haftalik_teknik_hisse_onerileri(html)
            documents.extend(haftalik_docs)
            logger.info(
                f"[{self.SOURCE_NAME}] Haftalik Teknik Hisse Önerileri: {len(haftalik_docs)} document(s)"
            )
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Haftalik Teknik Hisse Önerileri fetch failed: {exc}")

        return documents

    async def _extract_prices_from_teknik_image(self, image_url: str) -> list[dict[str, Any]]:
        """
        Extract expected prices from technical bulletin image via Ollama vision model (optional).
        Returns list of records:
          [{"hisse":"THYAO","beklenen_fiyat":"...","destek":"...","direnc":"...","zarar_kes":"..."}, ...]
        """
        if self._client is None:
            return []

        vision_model = getattr(settings, "ollama_vision_model", "").strip()
        if not vision_model:
            return []

        try:
            img_resp = await self._client.get(image_url)
            img_resp.raise_for_status()
            img_b64 = base64.b64encode(img_resp.content).decode("ascii")
        except Exception as exc:
            logger.warning(f"[{self.SOURCE_NAME}] Teknik görsel indirilemedi: {exc}")
            return []

        prompt = (
            "You are extracting Turkish technical bulletin table data from an image. "
            "Return ONLY valid JSON array. Each item must be: "
            "{\"hisse\":\"...\",\"beklenen_fiyat\":\"...\",\"destek\":\"...\",\"direnc\":\"...\",\"zarar_kes\":\"...\"}. "
            "If a field is missing, set it to null. No markdown."
        )
        payload = {
            "model": vision_model,
            "messages": [
                {"role": "user", "content": prompt, "images": [img_b64]},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=settings.ollama_timeout,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(content)
            if isinstance(parsed, list):
                cleaned: list[dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    hisse = str(item.get("hisse", "")).upper().strip()
                    if not re.fullmatch(r"[A-Z]{3,5}", hisse):
                        continue
                    cleaned.append(
                        {
                            "hisse": hisse,
                            "beklenen_fiyat": item.get("beklenen_fiyat"),
                            "destek": item.get("destek"),
                            "direnc": item.get("direnc"),
                            "zarar_kes": item.get("zarar_kes"),
                        }
                    )
                return cleaned
        except Exception as exc:
            logger.warning(f"[{self.SOURCE_NAME}] Teknik görsel OCR/VLM parse başarisiz: {exc}")
        return []

    async def _parse_gunluk_teknik_bulten(self, html: str) -> list[BulletinDocument]:
        soup = BeautifulSoup(html, "lxml")
        section = soup.select_one("#ContentSection .sub-page-content")
        if not section:
            return []

        title_h3 = section.find("h3")
        title = clean_ws(title_h3.get_text(" ", strip=True)) if title_h3 else "Teknik Bülten"
        doc_date = _parse_sabah_date_from_html(section) or _parse_any_date_from_html(section) or date.today()

        # Find technical chart/table image from editorfiles.
        image_url = ""
        for img in section.select("img[src]"):
            src = img.get("src", "")
            if "Pictures/editorfiles" in src:
                image_url = src if src.startswith("http") else self.BASE_URL + src
                break

        if not image_url:
            logger.warning(f"[{self.SOURCE_NAME}] Günlük Teknik Bülten görseli bulunamadi")
            return []

        records = await self._extract_prices_from_teknik_image(image_url)
        docs: list[BulletinDocument] = []

        if not records:
            # Fallback: keep a general document with image link for later processing.
            metin = (
                f"{title}. Teknik seviyeler görselde yer aliyor. "
                f"Görsel URL: {image_url}"
            )
            docs.append(
                BulletinDocument(
                    title=title,
                    stock_code="GENEL",
                    category="Gunluk_Teknik_Bulten",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.GUNLUK_TEKNIK_BULTEN_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "gunluk_teknik_bulten",
                        "image_url": image_url,
                        "ocr_source": "none",
                    },
                )
            )
            return docs

        for idx, rec in enumerate(records):
            hisse = rec.get("hisse", "GENEL")
            metin = (
                f"{hisse} için günlük teknik seviyeler: "
                f"beklenen_fiyat={rec.get('beklenen_fiyat')}, "
                f"destek={rec.get('destek')}, "
                f"direnc={rec.get('direnc')}, "
                f"zarar_kes={rec.get('zarar_kes')}."
            )
            docs.append(
                BulletinDocument(
                    title=f"{title} - {hisse} - {idx + 1}",
                    stock_code=hisse,
                    category="Gunluk_Teknik_Bulten",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.GUNLUK_TEKNIK_BULTEN_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "gunluk_teknik_bulten",
                        "image_url": image_url,
                        "ocr_source": "ollama_vision",
                        "beklenen_fiyat": rec.get("beklenen_fiyat"),
                        "destek": rec.get("destek"),
                        "direnc": rec.get("direnc"),
                        "zarar_kes": rec.get("zarar_kes"),
                        "mentioned_tickers": hisse,
                    },
                )
            )
        return docs

    async def _parse_haftalik_teknik_hisse_onerileri(self, html: str) -> list[BulletinDocument]:
        """
        Parse 'Haftalik Teknik Hisse Önerileri' page.
        Irregular publication; same editor-image pattern as daily technical bulletin.
        """
        soup = BeautifulSoup(html, "lxml")
        section = soup.select_one("#ContentSection .sub-page-content")
        if not section:
            return []

        title_h3 = section.find("h3")
        title = (
            clean_ws(title_h3.get_text(" ", strip=True))
            if title_h3
            else "Haftalik Teknik Hisse Önerileri"
        )
        doc_date = _parse_any_date_from_html(section) or date.today()

        image_url = ""
        for img in section.select("img[src]"):
            src = img.get("src", "")
            if "Pictures/editorfiles" in src:
                image_url = src if src.startswith("http") else self.BASE_URL + src
                break

        if not image_url:
            logger.warning(f"[{self.SOURCE_NAME}] Haftalik Teknik Hisse Önerileri görseli bulunamadi")
            return []

        records = await self._extract_prices_from_teknik_image(image_url)
        docs: list[BulletinDocument] = []

        if not records:
            metin = (
                f"{title}. Haftalik teknik seviyeler görselde yer aliyor. "
                f"Görsel URL: {image_url}"
            )
            docs.append(
                BulletinDocument(
                    title=title,
                    stock_code="GENEL",
                    category="Haftalik_Teknik_Hisse_Onerileri",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.HAFTALIK_TEKNIK_HISSE_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "haftalik_teknik_hisse_onerileri",
                        "yayin_sikligi": "duzensiz",
                        "image_url": image_url,
                        "ocr_source": "none",
                    },
                )
            )
            return docs

        for idx, rec in enumerate(records):
            hisse = rec.get("hisse", "GENEL")
            metin = (
                f"{hisse} için haftalik teknik seviyeler: "
                f"beklenen_fiyat={rec.get('beklenen_fiyat')}, "
                f"destek={rec.get('destek')}, "
                f"direnc={rec.get('direnc')}, "
                f"zarar_kes={rec.get('zarar_kes')}."
            )
            docs.append(
                BulletinDocument(
                    title=f"{title} - {hisse} - {idx + 1}",
                    stock_code=hisse,
                    category="Haftalik_Teknik_Hisse_Onerileri",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.HAFTALIK_TEKNIK_HISSE_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "haftalik_teknik_hisse_onerileri",
                        "yayin_sikligi": "duzensiz",
                        "image_url": image_url,
                        "ocr_source": "ollama_vision",
                        "beklenen_fiyat": rec.get("beklenen_fiyat"),
                        "destek": rec.get("destek"),
                        "direnc": rec.get("direnc"),
                        "zarar_kes": rec.get("zarar_kes"),
                        "mentioned_tickers": hisse,
                    },
                )
            )
        return docs

    async def _extract_rows_from_portfoy_image(self, image_url: str) -> list[dict[str, Any]]:
        """
        Extract portfolio rows from the image via optional Ollama vision model.
        Returns list of:
          [{"hisse":"...", "guncel_hisse_fiyati":"...", "hedef_hisse_fiyati":"...", "potansiyel_getiri":"...", "oneri":"..."}, ...]
        """
        if self._client is None:
            return []
        vision_model = getattr(settings, "ollama_vision_model", "").strip()
        if not vision_model:
            return []

        try:
            img_resp = await self._client.get(image_url)
            img_resp.raise_for_status()
            img_b64 = base64.b64encode(img_resp.content).decode("ascii")
        except Exception as exc:
            logger.warning(f"[{self.SOURCE_NAME}] Portföy görseli indirilemedi: {exc}")
            return []

        prompt = (
            "You are extracting Turkish equity portfolio table rows from an image. "
            "Return ONLY valid JSON array. "
            "Each row must include: "
            "{\"hisse\":\"...\",\"guncel_hisse_fiyati\":\"...\",\"hedef_hisse_fiyati\":\"...\",\"potansiyel_getiri\":\"...\",\"oneri\":\"...\"}. "
            "Use null for missing fields. No markdown, no prose."
        )
        payload = {
            "model": vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=settings.ollama_timeout,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                return []
            rows: list[dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                hisse = str(item.get("hisse", "")).upper().strip()
                if not re.fullmatch(r"[A-Z]{3,5}", hisse):
                    continue
                rows.append(
                    {
                        "hisse": hisse,
                        "guncel_hisse_fiyati": item.get("guncel_hisse_fiyati"),
                        "hedef_hisse_fiyati": item.get("hedef_hisse_fiyati"),
                        "potansiyel_getiri": item.get("potansiyel_getiri"),
                        "oneri": item.get("oneri"),
                    }
                )
            return rows
        except Exception as exc:
            logger.warning(f"[{self.SOURCE_NAME}] Portföy görsel OCR/VLM parse başarisiz: {exc}")
            return []

    async def _parse_hisse_oneri_portfoyu(self, html: str) -> list[BulletinDocument]:
        """
        Parse 'Hisse Öneri Portföyü' page.
        Note: this bulletin is not daily. We ingest latest published snapshot when available.
        """
        soup = BeautifulSoup(html, "lxml")
        section = soup.select_one("#ContentSection .sub-page-content")
        if not section:
            return []

        title_h3 = section.find("h3")
        title = clean_ws(title_h3.get_text(" ", strip=True)) if title_h3 else "Hisse Öneri Portföyü"
        doc_date = _parse_any_date_from_html(section) or date.today()

        image_url = ""
        for img in section.select("img[src]"):
            src = img.get("src", "")
            if "Pictures/editorfiles" in src:
                image_url = src if src.startswith("http") else self.BASE_URL + src
                break
        if not image_url:
            logger.warning(f"[{self.SOURCE_NAME}] Hisse Öneri Portföyü görseli bulunamadi")
            return []

        rows = await self._extract_rows_from_portfoy_image(image_url)
        docs: list[BulletinDocument] = []

        if not rows:
            metin = (
                f"{title}. Portföy detaylari görselde yer aliyor. "
                f"Görsel URL: {image_url}"
            )
            docs.append(
                BulletinDocument(
                    title=title,
                    stock_code="GENEL",
                    category="Hisse_Oneri_Portfoyu",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.HISSE_ONERI_PORTFOYU_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "hisse_oneri_portfoyu",
                        "yayin_sikligi": "duzensiz",
                        "image_url": image_url,
                        "ocr_source": "none",
                    },
                )
            )
            return docs

        for idx, row in enumerate(rows):
            hisse = row.get("hisse", "GENEL")
            metin = (
                f"{hisse} portföy önerisi: "
                f"guncel_hisse_fiyati={row.get('guncel_hisse_fiyati')}, "
                f"hedef_hisse_fiyati={row.get('hedef_hisse_fiyati')}, "
                f"potansiyel_getiri={row.get('potansiyel_getiri')}, "
                f"oneri={row.get('oneri')}."
            )
            docs.append(
                BulletinDocument(
                    title=f"{title} - {hisse} - {idx + 1}",
                    stock_code=hisse,
                    category="Hisse_Oneri_Portfoyu",
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.HISSE_ONERI_PORTFOYU_URL,
                    extra_metadata={
                        "kurum": KURUM_BY_SOURCE.get(self.SOURCE_NAME, self.SOURCE_NAME),
                        "bulten_turu": "hisse_oneri_portfoyu",
                        "yayin_sikligi": "duzensiz",
                        "image_url": image_url,
                        "ocr_source": "ollama_vision",
                        "guncel_hisse_fiyati": row.get("guncel_hisse_fiyati"),
                        "hedef_hisse_fiyati": row.get("hedef_hisse_fiyati"),
                        "potansiyel_getiri": row.get("potansiyel_getiri"),
                        "oneri": row.get("oneri"),
                        "mentioned_tickers": hisse,
                    },
                )
            )
        return docs



    def _parse_sabah_stratejisi(self, html: str, source_url: str = "") -> list[BulletinDocument]:
        soup = BeautifulSoup(html, "lxml")
        raw_fragment, title = _extract_main_content_html(soup)
        if not raw_fragment or len(raw_fragment) < 200:
            loc = f" — {source_url}" if source_url else ""
            logger.warning(
                f"[{self.SOURCE_NAME}] Sabah Stratejisi: içerik bloğu çok kisa veya bulunamadi "
                f"(#ContentSection / .sub-page-content kontrol edin){loc}"
            )
            return []

        doc_date = _parse_sabah_date_from_html(BeautifulSoup(raw_fragment, "lxml")) or date.today()
        pdf_url = ""
        for a in BeautifulSoup(raw_fragment, "lxml").find_all("a", href=True):
            href = a["href"]
            if "Documents" in href and href.lower().endswith(".pdf"):
                pdf_url = href if href.startswith("http") else self.BASE_URL + href
                break
        mentioned_tickers = _extract_mentioned_tickers(raw_fragment)

        records = _extract_section_records(raw_fragment)
        documents: list[BulletinDocument] = []

        if not records:
            # Fallback: keep old behavior as a single generic document.
            documents.append(
                BulletinDocument(
                    title=title or "Sabah Stratejisi",
                    stock_code="GENEL",
                    category="Sabah Stratejisi",
                    raw_html=raw_fragment,
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.SABAH_STRATEJISI_URL,
                    extra_metadata={
                        "bulten_turu": "sabah_stratejisi",
                        "pdf_url": pdf_url,
                        "mentioned_tickers": ",".join(mentioned_tickers),
                    },
                )
            )
            return documents

        # Preferred behavior: one semantic BulletinDocument per parsed record.
        for idx, rec in enumerate(records):
            meta = rec["metadata"]
            metin = rec["metin"]
            hisse = meta.get("hisse", "GENEL")
            kategori = meta.get("kategori", "Genel")
            sirket = meta.get("sirket", "")

            if hisse and hisse != "GENEL":
                mentioned_for_doc = hisse
            elif hisse == "GENEL":
                mentioned_for_doc = ",".join(mentioned_tickers)
            else:
                # Tickerless institution (e.g., Ziraat Bankasi:) — do not copy full bulletin ticker list
                mentioned_for_doc = ""

            documents.append(
                BulletinDocument(
                    title=f"{title} - {sirket or hisse or 'Kayit'} - {idx + 1}",
                    stock_code=hisse,
                    category=kategori,
                    raw_html=wrap_metin_paragraph(metin),
                    date=doc_date,
                    source=self.SOURCE_NAME,
                    url=self.SABAH_STRATEJISI_URL,
                    extra_metadata={
                        "bulten_turu": "sabah_stratejisi",
                        "pdf_url": pdf_url,
                        "sirket": sirket,
                        "mentioned_tickers": mentioned_for_doc,
                    },
                )
            )

        return documents
