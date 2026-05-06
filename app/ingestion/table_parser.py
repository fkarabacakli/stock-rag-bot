"""
HTML table parser → structured JSON.

Converts brokerage-report tables (support/resistance, stop-loss levels,
financials) into JSON strings that can be embedded alongside text chunks.
"""
from __future__ import annotations

import io
import json
import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

# Column name normalization map (Turkish → canonical key)
_COLUMN_ALIASES: dict[str, str] = {
    # Price levels
    "destek": "destek",
    "direnç": "direnc",
    "direnç 1": "direnc_1",
    "direnç 2": "direnc_2",
    "destek 1": "destek_1",
    "destek 2": "destek_2",
    "zarar kes": "zarar_kes",
    "stop loss": "zarar_kes",
    "stop-loss": "zarar_kes",
    "hedef fiyat": "hedef_fiyat",
    "hedef": "hedef_fiyat",
    "target": "hedef_fiyat",
    # Financial metrics
    "fiyat": "fiyat",
    "price": "fiyat",
    "f/k": "fk_orani",
    "pd/dd": "pddd",
    "favök": "favok",
    "ebitda": "favok",
    "net kar": "net_kar",
    "ciro": "ciro",
    # Recommendation
    "öneri": "oneri",
    "tavsiye": "oneri",
    "recommendation": "oneri",
    "getiri": "getiri_potansiyeli",
    "potansiyel": "getiri_potansiyeli",
}


def _normalize_col(name: str) -> str:
    """Normalize a column header to a canonical key."""
    cleaned = name.strip().lower()
    return _COLUMN_ALIASES.get(cleaned, re.sub(r"\s+", "_", cleaned))


def _parse_single_table(table_tag) -> Optional[dict]:
    """Convert a single <table> BS4 tag into a structured dict."""
    try:
        df = pd.read_html(io.StringIO(str(table_tag)))[0]
    except Exception:
        return None

    if df.empty:
        return None

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

    df.columns = [_normalize_col(str(c)) for c in df.columns]

    # Drop completely empty rows/cols
    df = df.dropna(how="all").dropna(axis=1, how="all")

    records = df.to_dict(orient="records")
    # Clean NaN values
    clean_records = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = ""
            elif isinstance(v, float) and v == int(v):
                clean_row[k] = str(int(v))
            else:
                clean_row[k] = str(v)
        clean_records.append(clean_row)

    return {"columns": list(df.columns), "rows": clean_records}


def extract_tables_from_html(html: str) -> list[dict]:
    """
    Extract all tables from raw HTML and return a list of structured dicts.
    Each dict has 'columns' and 'rows' keys.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    results = []
    for i, table in enumerate(tables):
        parsed = _parse_single_table(table)
        if parsed:
            parsed["table_index"] = i
            results.append(parsed)
        else:
            logger.debug(f"[table_parser] Skipped empty table at index {i}")

    return results


def tables_to_text(tables: list[dict]) -> str:
    """
    Convert parsed tables to a human-readable text representation
    suitable for embedding in the vector store.
    """
    if not tables:
        return ""

    lines = []
    for t in tables:
        cols = t.get("columns", [])
        rows = t.get("rows", [])
        lines.append("=== Tablo ===")
        lines.append(" | ".join(cols))
        lines.append("-" * 40)
        for row in rows:
            row_vals = [str(row.get(c, "")) for c in cols]
            lines.append(" | ".join(row_vals))
        lines.append("")

    return "\n".join(lines)


def tables_to_json_string(tables: list[dict]) -> str:
    """Serialize parsed tables to a compact JSON string for metadata storage."""
    return json.dumps(tables, ensure_ascii=False, default=str)


def parse_and_format_tables(html: str) -> tuple[str, str]:
    """
    Parse all tables in HTML.

    Returns:
        (human_readable_text, json_string)
    """
    tables = extract_tables_from_html(html)
    text = tables_to_text(tables)
    json_str = tables_to_json_string(tables)
    return text, json_str
