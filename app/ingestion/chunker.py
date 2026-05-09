"""
HTML-tag-based intelligent chunker.

Strategy:
  - <h3> tags → section boundaries (new chunk)
  - <strong> tags → key metadata extraction (support/resistance levels, targets)
  - Each chunk carries the parent document's metadata plus section-level context
  - Minimum chunk length enforced to avoid noise vectors
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger

from app.ingestion.scrapers.base import BulletinDocument

_MIN_CHUNK_CHARS = 80
_MAX_CHUNK_CHARS = 1500
# For single-<p> long content (e.g., Sabah company news): split by sentence first, then buffer
_MAX_CHARS_PER_PARA_PIECE = 550
_PARAGRAPH_BUFFER_MAX = 700


@dataclass
class Chunk:
    """A semantic chunk ready for embedding."""

    text: str
    doc_id: str
    chunk_idx: int
    section_title: Optional[str]
    metadata: dict = field(default_factory=dict)
    strong_keys: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Prepend section title + strong keys to boost semantic signal."""
        parts = []
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.strong_keys:
            parts.append("Key: " + " | ".join(self.strong_keys))
        parts.append(self.text)
        return "\n".join(parts)

    def chromadb_id(self) -> str:
        return f"{self.doc_id}_chunk{self.chunk_idx}"


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0100-\u017E\u0130-\u0131\u00E7-\u00FC\u011F\u015F]", "", text)
    return text.strip()


def _extract_strong_values(tag: Tag) -> list[str]:
    """Extract text from <strong> tags within a section — typically key levels."""
    return [
        _clean_text(s.get_text())
        for s in tag.find_all("strong")
        if _clean_text(s.get_text())
    ]


def _split_long_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text exceeding max_chars at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    if not parts:
        return [text[:max_chars]]
    # No sentence boundaries (or single huge sentence): hard-split any oversized part
    final: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            final.append(p)
        else:
            for i in range(0, len(p), max_chars):
                final.append(p[i : i + max_chars])
    return final


def _stable_doc_id(doc: BulletinDocument) -> str:
    """
    Prevent Chroma ID collisions across multiple documents on the same day/stock code.
    Use a hash of url + title + category + content summary for uniqueness.
    """
    body = (doc.raw_html or "")[:6000]
    key = f"{doc.url}|{doc.title}|{doc.category}|{body}"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc.source}_{doc.date.isoformat()}_{doc.stock_code}_{suffix}"


def chunk_document(doc: BulletinDocument) -> list[Chunk]:
    """
    Parse a BulletinDocument's raw_html and return a list of Chunks.

    Chunking logic:
      1. <h3> tags split the HTML into sections.
      2. Each section's text is extracted and cleaned.
      3. <strong> values are noted as key signals.
      4. Very long sections are further split at sentence boundaries.
    """
    soup = BeautifulSoup(doc.raw_html, "lxml")
    doc_id = _stable_doc_id(doc)
    base_metadata = doc.to_metadata()

    chunks: list[Chunk] = []
    chunk_idx = 0

    # Strategy 1: Split on <h3> section headers
    h3_sections = soup.find_all("h3")

    if h3_sections:
        for h3 in h3_sections:
            section_title = _clean_text(h3.get_text())
            # Collect sibling content until next h3
            section_content_tags = []
            sibling = h3.find_next_sibling()
            while sibling and sibling.name != "h3":
                section_content_tags.append(sibling)
                sibling = sibling.find_next_sibling() if hasattr(sibling, "find_next_sibling") else None

            # Build a temporary container to extract text + strong keys
            section_html = "".join(str(t) for t in section_content_tags)
            section_soup = BeautifulSoup(section_html, "lxml")
            section_text = _clean_text(section_soup.get_text(separator=" "))
            strong_keys = _extract_strong_values(section_soup)

            if len(section_text) < _MIN_CHUNK_CHARS:
                continue

            for text_part in _split_long_text(section_text):
                chunk = Chunk(
                    text=text_part,
                    doc_id=doc_id,
                    chunk_idx=chunk_idx,
                    section_title=section_title,
                    metadata={**base_metadata, "section": section_title},
                    strong_keys=strong_keys,
                )
                chunks.append(chunk)
                chunk_idx += 1

    # Strategy 2: No <h3> found — split on <p> or plain paragraphs
    if not chunks:
        paragraphs = soup.find_all("p")
        if not paragraphs:
            # Last resort: split full text by double newlines
            full_text = _clean_text(soup.get_text(separator=" "))
            paragraphs_text = [p.strip() for p in full_text.split("  ") if p.strip()]
        else:
            paragraphs_text = [_clean_text(p.get_text(separator=" ")) for p in paragraphs]

        # If a single <p> has 1000+ chars (Ziraat company summary), split by sentences first
        expanded_paras: list[str] = []
        for raw in paragraphs_text:
            if not raw or len(raw) < 20:
                continue
            if len(raw) > _MAX_CHARS_PER_PARA_PIECE:
                expanded_paras.extend(
                    _split_long_text(raw, max_chars=_MAX_CHARS_PER_PARA_PIECE)
                )
            else:
                expanded_paras.append(raw)

        buffer = ""
        strong_keys = _extract_strong_values(soup)

        for para in expanded_paras:
            if len(buffer) + len(para) + (1 if buffer else 0) <= _PARAGRAPH_BUFFER_MAX:
                buffer = (buffer + " " + para).strip() if buffer else para
            else:
                if len(buffer) >= _MIN_CHUNK_CHARS:
                    chunks.append(Chunk(
                        text=buffer,
                        doc_id=doc_id,
                        chunk_idx=chunk_idx,
                        section_title=None,
                        metadata={**base_metadata},
                        strong_keys=strong_keys,
                    ))
                    chunk_idx += 1
                buffer = para

        if len(buffer) >= _MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                text=buffer,
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                section_title=None,
                metadata={**base_metadata},
                strong_keys=strong_keys,
            ))

    logger.debug(
        f"[chunker] {doc.source}/{doc.stock_code} → {len(chunks)} chunks"
    )
    return chunks


def chunk_documents(docs: list[BulletinDocument]) -> list[Chunk]:
    """Chunk a list of documents."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
