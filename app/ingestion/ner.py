"""
Financial NER — lightweight regex + confidence-ranked BIST ticker extraction.

Detection priority (high → low):
  1. Parenthetical  "(THYAO)"              → unambiguous, always first
  2. Line-start     "THYAO: …" / "THYAO - "→ company-as-subject
  3. Inline UPPER   any standalone TOKEN    → medium confidence

API
---
  extract_tickers(text, scan_chars)  → list[str]   ordered, deduplicated
  primary_ticker(tickers)            → str          first or "GENEL"
  tickers_to_str(tickers)            → str          comma-separated for metadata
"""
from __future__ import annotations

import re

# ── Patterns ───────────────────────────────────────────────────────────────────

# "(THYAO)" or "(MARGUN, Nötr)"
_RE_PAREN = re.compile(r"\(([A-Z]{3,6})[,\s\)]")

# "THYAO:" or "MARGUN -" at the very start of a line
_RE_LINE_START = re.compile(r"(?m)^([A-Z]{3,6})\s*[:\-\–]")

# Any standalone 3-6 uppercase ASCII letter word
_RE_ANY = re.compile(r"\b([A-Z]{3,6})\b")

# ── Blacklist ──────────────────────────────────────────────────────────────────

_BLACKLIST: frozenset[str] = frozenset({
    # Geopolitical / supranational
    "ABD", "FED", "ECB", "IMF", "WHO", "NATO", "OECD", "BOJ", "BOE",
    "AMB", "SEC", "BIS", "WTO", "FBI", "CIA",
    # Currencies & benchmark indices
    "USD", "EUR", "TRY", "TL", "GBP", "JPY", "CHF",
    "BIST", "FTSE", "MSCI", "SPX", "VIX", "ETF", "IPO", "NYSE", "NDAQ",
    # TR macro / regulatory bodies
    "TCMB", "PPK", "BDDK", "SPK",
    "PMI", "CPI", "PPI", "PCE", "ISM", "ADP", "MBA",
    "TUFE", "UFE", "GSYH", "TUIK", "KDV", "OTV",
    "TÜFE", "ÜFE", "GSYİH", "TÜİK",
    # Generic financial abbreviations
    "CEO", "CFO", "COO", "CRO", "IPO",
    "YOY", "QOQ", "TTM", "EPS", "ROE", "ROA", "NPL",
    # Report / document jargon
    "PDF", "HLY", "HY", "GYO", "IHH",
    # Misc country / region codes
    "TR", "TUR", "EU", "UK", "ABD",
    # Common false positives seen in Halk bulletins
    "BIST", "BİST", "VERI", "TABLO",
})


def extract_tickers(text: str, scan_chars: int = 600) -> list[str]:
    """
    Extract BIST ticker candidates from the first `scan_chars` characters of *text*.

    Returns a deduplicated list ordered by detection confidence:
    parenthetical matches come first, then line-start matches, then inline.

    Example
    -------
    >>> extract_tickers("THYAO (THYAO): Yolcu sayısı artışı. GARAN da etkilendi.")
    ['THYAO', 'GARAN']
    """
    snippet = text[:scan_chars]
    seen: set[str] = set()
    result: list[str] = []

    def _push(raw: str) -> None:
        t = raw.strip().upper()
        if t and len(t) >= 3 and t not in _BLACKLIST and t not in seen:
            seen.add(t)
            result.append(t)

    for m in _RE_PAREN.finditer(snippet):       # Priority 1
        _push(m.group(1))
    for m in _RE_LINE_START.finditer(snippet):  # Priority 2
        _push(m.group(1))
    for m in _RE_ANY.finditer(snippet):         # Priority 3
        _push(m.group(1))

    return result


def primary_ticker(tickers: list[str]) -> str:
    """Return the highest-confidence ticker, or 'GENEL' when none found."""
    return tickers[0] if tickers else "GENEL"


def tickers_to_str(tickers: list[str]) -> str:
    """Return comma-separated ticker string suitable for metadata storage."""
    return ",".join(tickers) if tickers else ""
