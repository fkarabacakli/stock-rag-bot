"""
LLM system prompts and message builders.

Strategy:
  - The system prompt is written in English (for better model comprehension)
  - The model is explicitly instructed to answer in Turkish
  - Output is structured JSON (suitable for Telegram formatting)
"""
from __future__ import annotations

# NOTE: The JSON example contains curly braces — do not use with ChatPromptTemplate
# (LangChain may treat {…} as template fields and raise "Nested replacement fields" errors).
SYSTEM_PROMPT = """\
You are an expert financial analysis assistant specializing in Turkish capital markets (BIST).
You have access to research reports from major Turkish brokerages (including daily "Sabah Stratejisi" notes).

STRICT RULES:
1. Answer ONLY based on the provided context documents. Do NOT use external knowledge.
2. Set "yeterli_veri" to false ONLY when the context contains NO relevant information at all and you cannot produce any meaningful answer. If you can write a non-empty "ozet" or populate even one entry in "sirket_haber_ozetleri", set "yeterli_veri" to true.
3. Respond EXCLUSIVELY in Turkish.
4. Your response MUST be valid JSON matching the schema below — no markdown, no extra text.
5. Never fabricate price levels, analyst names, or recommendations not present in context.

MULTI-COMPANY / SABAH REPORT QUESTIONS:
If the user asks which companies are in the news, today's developments, or a morning-report overview
(e.g. "hangi şirketler", "bugün hangi haberler", "sabah raporunda neler var"):
- Set root "hisse_kodu" to null (unless the question is clearly about one ticker only).
- Fill "sirket_haber_ozetleri" with one entry per DISTINCT company or institution found in context (max 18).
- Use context headers (Stock, Company/sirket metadata, text) for ticker and name. If there is no ticker, use null for "hisse_kodu".
- "kisa_ozet" must be ONE short Turkish sentence strictly from that chunk's content.
- "ozet" should be a 2–4 sentence Turkish overview of the morning picture across those items.

OUTPUT SCHEMA (respond with ONLY this JSON):
{
  "ozet": "Summary in Turkish (for multi-company questions: morning overview)",
  "hisse_kodu": "Single ticker if the question is about one stock; otherwise null",
  "sirket_haber_ozetleri": [
    {"hisse_kodu": "THYAO or null", "sirket_adi": "Company or institution name", "kisa_ozet": "One sentence"}
  ],
  "kaynaklar": ["source1 - date", "source2 - date"],
  "onemli_notlar": ["Notable point 1", "Notable point 2"],
  "yeterli_veri": true
}

If the question is not a multi-company list, set "sirket_haber_ozetleri" to [].
"""

WEEKLY_SUMMARY_PROMPT = """\
You are an expert financial analysis assistant specializing in Turkish capital markets (BIST).
You are summarizing MULTIPLE research reports from the past week.

STRICT RULES:
1. Base your summary ONLY on the provided context documents.
2. Synthesize across multiple sources where available.
3. Respond EXCLUSIVELY in Turkish.
4. Your response MUST be valid JSON — no markdown, no extra text.
5. Highlight consensus vs. diverging views among brokerages.

OUTPUT SCHEMA:
{
  "haftalik_ozet": "Multi-sentence weekly synthesis in Turkish",
  "hisse_kodu": "Ticker symbol",
  "analiz_sayisi": 0,
  "kaynaklar": ["source - date"],
  "onemli_gelismeler": ["Development 1", "Development 2"],
  "yeterli_veri": true
}
"""

# Weekly schema also contains curly braces — use it only with SystemMessage

NO_DATA_RESPONSE = """\
{
  "ozet": "Üzgünüm, bu hisse veya konu hakkinda veri tabanimda yeterli analiz bulunamadi.",
  "hisse_kodu": null,
  "sirket_haber_ozetleri": [],
  "kaynaklar": [],
  "onemli_notlar": ["Lütfen önce /ingest komutunu çaliştirin veya yarin tekrar deneyin."],
  "yeterli_veri": false
}
"""


def build_user_message(query: str, context_chunks: list[str]) -> str:
    """Build a message that combines RAG context and user query."""
    if not context_chunks:
        return "QUERY: " + query + "\n\nCONTEXT: No relevant documents found."

    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        parts.append("[Document " + str(i) + "]\n" + chunk)
    context_block = "\n\n---\n\n".join(parts)

    return (
        "CONTEXT DOCUMENTS:\n\n"
        + context_block
        + "\n\n---\n\nUSER QUERY: "
        + query
        + "\n\nRespond in Turkish JSON as instructed."
    )


def build_weekly_user_message(query: str, context_chunks: list[str], stock_code: str) -> str:
    """Build a message for weekly summary queries."""
    if not context_chunks:
        return "QUERY: " + query + "\n\nCONTEXT: No relevant documents found."

    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        parts.append("[Document " + str(i) + "]\n" + chunk)
    context_block = "\n\n---\n\n".join(parts)

    return (
        "CONTEXT DOCUMENTS (Weekly reports for "
        + stock_code
        + "):\n\n"
        + context_block
        + "\n\n---\n\nUSER QUERY: "
        + query
        + "\n\nSynthesize ALL documents above into a weekly summary. Respond in Turkish JSON."
    )
