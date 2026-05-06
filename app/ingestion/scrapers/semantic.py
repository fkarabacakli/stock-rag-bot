"""Shared helpers for semantic bulletin records (metin + metadata)."""
from __future__ import annotations

import html
import re

# Human-readable brokerage names for vector metadata / API consumers
KURUM_BY_SOURCE: dict[str, str] = {
    "ziraat_yatirim": "Ziraat Yatırım",
}


def slug_category_tr(label: str) -> str:
    """Turn Turkish category labels into stable snake-ish tokens."""
    s = " ".join(label.split()).strip()
    mapping = {
        "Genel Analiz": "Genel_Analiz",
        "Teknik Analiz": "Teknik_Analiz",
        "Temel Analiz": "Temel_Analiz",
        "Strateji": "Strateji",
        "Günlük Bülten": "Gunluk_Bulten",
        "Haftalık Bülten": "Haftalik_Bulten",
        "Model Portföy": "Model_Portfoy",
        "Yatırım Tavsiyesi": "Yatirim_Tavsiyesi",
        "Hedef Fiyat Revizyonu": "Hedef_Fiyat_Revizyonu",
    }
    if s in mapping:
        return mapping[s]
    if not s:
        return "Genel"
    return s.replace(" ", "_")


def wrap_metin_paragraph(metin: str) -> str:
    """Minimal HTML wrapper for downstream chunker / table parser."""
    return f"<p>{html.escape(metin)}</p>"


def clean_ws(text: str) -> str:
    return " ".join(text.split()).strip()


_TICKER_IN_PARENS = re.compile(r"\b([A-Z]{3,5})\b")


def ticker_from_parenthetical(paren: str) -> str:
    """First plausible BIST ticker inside parentheses segment, e.g. 'ARCLK, Nötr'."""
    paren_u = paren.upper().strip()
    first = paren_u.split(",")[0].strip()
    if re.fullmatch(r"[A-Z]{3,5}", first):
        return first
    m = _TICKER_IN_PARENS.search(paren_u)
    return m.group(1) if m else "GENEL"
