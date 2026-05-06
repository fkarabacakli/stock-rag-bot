#!/usr/bin/env python3
"""
Manuel metadata / chunk kontrolü — Chroma veya embedding çalıştırmaz.

Proje kökünden:
  python scripts/preview_ingestion_metadata.py
  python scripts/preview_ingestion_metadata.py --scraper ziraat --limit 3
  python scripts/preview_ingestion_metadata.py --out ./preview/son_inceleme.json
  python scripts/preview_ingestion_metadata.py --full-text --limit 1

Çıktı: metadata + chunk alanları; --full-text ile raw_html, text, embedding_text kesilmeden.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from app.ingestion.chunker import chunk_document
except ModuleNotFoundError as exc:
    print(
        f"\n[preview] Modül bulunamadı: {exc.name}\n"
        "Bu script, bot/uvicorn çalıştırdığın Python ile aynı ortamda çalışmalı.\n"
        "Kontrol:  which python && python -c \"import bs4\"\n"
        "Kurulum (proje kökünde):\n"
        "  pip install -r requirements.txt\n"
        "veya sadece:\n"
        "  pip install beautifulsoup4 lxml\n",
        file=sys.stderr,
    )
    sys.exit(1)

from app.ingestion.pipeline import ACTIVE_SCRAPERS
from app.ingestion.scrapers.base import BulletinDocument
from app.ingestion.scrapers.ziraat_yatirim import ZiraatYatirimScraper

# Önizleme: --scraper ziraat | all (all = pipeline'daki tüm ACTIVE_SCRAPERS)
SCRAPER_BY_NAME = {"ziraat": ZiraatYatirimScraper}


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"... [{len(s)} karakter]"


def _doc_to_preview_dict(
    doc: BulletinDocument, html_preview_len: int, *, full_text: bool
) -> dict:
    base = {
        "title": doc.title,
        "stock_code": doc.stock_code,
        "category": doc.category,
        "date": doc.date.isoformat(),
        "source": doc.source,
        "url": doc.url,
        "extra_metadata": doc.extra_metadata,
        "to_metadata()": doc.to_metadata(),
        "raw_html_length": len(doc.raw_html or ""),
    }
    if full_text:
        base["raw_html"] = doc.raw_html or ""
    else:
        base["raw_html_preview"] = _truncate(doc.raw_html or "", html_preview_len)
    return base


def _chunk_to_preview_dict(c, text_preview_len: int, *, full_text: bool) -> dict:
    base = {
        "chromadb_id": c.chromadb_id(),
        "doc_id": c.doc_id,
        "chunk_idx": c.chunk_idx,
        "section_title": c.section_title,
        "strong_keys": c.strong_keys,
        "metadata": dict(c.metadata),
    }
    if full_text:
        base["text"] = c.text
        base["embedding_text"] = c.embedding_text
    else:
        base["text_preview"] = _truncate(c.text, text_preview_len)
        base["embedding_text_preview"] = _truncate(c.embedding_text, text_preview_len)
    return base


async def _fetch_docs(which: str) -> list[BulletinDocument]:
    if which == "all":
        parts = await asyncio.gather(
            *[cls().safe_fetch_bulletins() for cls in ACTIVE_SCRAPERS]
        )
        out: list[BulletinDocument] = []
        for p in parts:
            out.extend(p)
        return out
    cls = SCRAPER_BY_NAME[which]
    return await cls().safe_fetch_bulletins()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape sonrası metadata ve chunk önizlemesi (manuel kontrol)"
    )
    parser.add_argument(
        "--scraper",
        choices=["all", "ziraat"],
        default="all",
        help="all = pipeline ACTIVE_SCRAPERS; şimdilik yalnızca Ziraat",
    )
    parser.add_argument("--limit", type=int, default=0, help="İlk N belge (0=tümü)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON çıktı (varsayılan: preview/metadata_preview_<bugün>.json)",
    )
    parser.add_argument(
        "--html-preview",
        type=int,
        default=600,
        help="Belge HTML önizleme karakter sayısı",
    )
    parser.add_argument(
        "--text-preview",
        type=int,
        default=500,
        help="Chunk metin önizleme karakter sayısı",
    )
    parser.add_argument(
        "--dump-html-dir",
        type=Path,
        default=None,
        help="İsteğe bağlı: her belgenin tam HTML'ini bu klasöre yaz",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="JSON'da raw_html, chunk text ve embedding_text tam (kesme yok)",
    )
    args = parser.parse_args()

    async def run() -> Path:
        print("[preview] Kazıma başlıyor...", flush=True)
        docs = await _fetch_docs(args.scraper)
        if args.limit > 0:
            docs = docs[: args.limit]
        print(f"[preview] {len(docs)} belge alındı.", flush=True)

        out_path = args.out
        if out_path is None:
            preview_dir = _ROOT / "preview"
            preview_dir.mkdir(exist_ok=True)
            out_path = preview_dir / f"metadata_preview_{date.today().isoformat()}.json"

        payload_docs = []
        total_chunks = 0

        if args.dump_html_dir:
            args.dump_html_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(docs):
            safe_name = f"{i:03d}_{doc.source}_{doc.stock_code}".replace("/", "_")
            chunks = chunk_document(doc)
            total_chunks += len(chunks)

            entry = {
                "index": i,
                "document": _doc_to_preview_dict(
                    doc, args.html_preview, full_text=args.full_text
                ),
                "chunks": [
                    _chunk_to_preview_dict(c, args.text_preview, full_text=args.full_text)
                    for c in chunks
                ],
            }
            if args.dump_html_dir:
                html_path = args.dump_html_dir / f"{safe_name}.html"
                html_path.write_text(doc.raw_html or "", encoding="utf-8")
                entry["dumped_html_file"] = str(html_path)

            payload_docs.append(entry)

            # Konsola kısa özet
            print(
                f"\n--- [{i}] {doc.category} | {doc.stock_code} | chunks={len(chunks)} ---",
                flush=True,
            )
            print(f"    url: {doc.url}", flush=True)
            meta = doc.to_metadata()
            print(f"    metadata keys: {list(meta.keys())}", flush=True)
            for j, ch in enumerate(chunks[:3]):
                print(f"    chunk[{j}] id={ch.chromadb_id()} section={ch.section_title!r}", flush=True)
            if len(chunks) > 3:
                print(f"    ... +{len(chunks) - 3} chunk daha", flush=True)

        payload = {
            "generated_at": date.today().isoformat(),
            "scraper": args.scraper,
            "document_count": len(docs),
            "chunk_count": total_chunks,
            "documents": payload_docs,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[preview] Tam JSON yazıldı: {out_path}", flush=True)
        if args.full_text:
            print(
                "[preview] --full-text: dosya büyük olabilir; terminalde jq/less ile aç.",
                flush=True,
            )
        return out_path

    asyncio.run(run())


if __name__ == "__main__":
    main()
