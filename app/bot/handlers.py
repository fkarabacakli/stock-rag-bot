"""
Telegram bot command and message handlers.

Commands:
  /start         — Welcome message + main menu
  /analiz <KOD>  — Single stock analysis
  /haftalik <KOD>— Weekly synthesis for a stock
  /kurumlar      — Choose brokerage source
  /model         — Switch LLM model
  /ingest        — Manually trigger ingestion (admin)
  /durum         — System health status
  <free text>    — Free-form RAG query
"""
from __future__ import annotations

import html
import re
from typing import Optional

from loguru import logger
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.bot.keyboards import (
    analysis_type_keyboard,
    main_menu_keyboard,
    model_selection_keyboard,
    source_selection_keyboard,
    stock_quick_access_keyboard,
)
from app.config import get_settings

_settings = get_settings()

# Per-user session state stored in context.user_data
_SESSION_MODEL_KEY = "selected_model"
_SESSION_SOURCE_KEY = "selected_source"

WELCOME_MESSAGE = (
    "Merhaba! Ben *Finansal RAG Asistanınızım*.\n\n"
    "Ziraat Yatırım bültenlerini (Sabah Stratejisi dahil) analiz ediyorum.\n\n"
    "*Komutlar:*\n"
    "/analiz THYAO — Hisse analizi\n"
    "/haftalik GARAN — Haftalık özet\n"
    "/kurumlar — Kaynak kurum seçimi\n"
    "/model — LLM model seçimi\n"
    "/durum — Sistem durumu\n\n"
    "Sabah raporunda *hangi şirketler* var, *bugün hangi gelişmeler* var gibi soruları "
    "doğrudan yazabilir veya aşağıdaki *Bugün hangi şirketler?* düğmesine basabilirsiniz."
)


def _format_rag_response(data: dict) -> str:
    """
    RAG JSON yanıtını Telegram HTML (ParseMode.HTML) için biçimlendirir.
    LLM kaynaklı tüm metin html.escape ile sarılır; entity parse hatası oluşmaz.
    """
    e = html.escape

    yeterli = data.get("yeterli_veri", True)
    if not yeterli:
        ozet = data.get("ozet") or "Yeterli veri bulunamadı."
        return f"⚠️ <b>Yetersiz Veri</b>\n\n{e(str(ozet))}"

    lines: list[str] = []

    ticker = data.get("hisse_kodu") or ""
    if ticker:
        lines.append(f"📊 <b>{e(str(ticker))} Analizi</b>")
        lines.append("")

    ozet = data.get("ozet") or ""
    if ozet:
        lines.append(f"<b>Özet:</b>\n{e(str(ozet))}")
        lines.append("")

    haftalik = data.get("haftalik_ozet") or ""
    if haftalik:
        lines.append(f"<b>Haftalık özet:</b>\n{e(str(haftalik))}")
        lines.append("")

    konsensus = data.get("konsensus_oneri")
    if konsensus:
        lines.append(f"<b>Konsensus:</b> {e(str(konsensus))}")
        lines.append("")

    sirketler = data.get("sirket_haber_ozetleri") or []
    if sirketler:
        lines.append("<b>Şirket / kurum haberleri:</b>")
        for item in sirketler[:18]:
            if not isinstance(item, dict):
                lines.append(f"  • {e(str(item))}")
                continue
            kod = item.get("hisse_kodu") or "—"
            ad = item.get("sirket_adi") or ""
            kisa = item.get("kisa_ozet") or ""
            label = f"{kod}" + (f" ({ad})" if ad else "")
            lines.append(f"  • <b>{e(str(label))}</b> — {e(str(kisa))}")
        lines.append("")

    seviyeler = data.get("seviyeler") or {}
    if isinstance(seviyeler, dict) and seviyeler:
        weekly_keys = ("destek_araligi", "direnc_araligi", "hedef_fiyat_ort")
        if any(seviyeler.get(k) for k in weekly_keys):
            lines.append("<b>Seviye özeti:</b>")
            if seviyeler.get("destek_araligi"):
                lines.append(f"  🟢 Destek: {e(str(seviyeler['destek_araligi']))}")
            if seviyeler.get("direnc_araligi"):
                lines.append(f"  🔴 Direnç: {e(str(seviyeler['direnc_araligi']))}")
            if seviyeler.get("hedef_fiyat_ort"):
                lines.append(f"  🎯 Hedef (ort.): {e(str(seviyeler['hedef_fiyat_ort']))}")
            lines.append("")
        elif (
            seviyeler.get("hedef_fiyat")
            or seviyeler.get("zarar_kes")
            or seviyeler.get("destek")
            or seviyeler.get("direnc")
        ):
            lines.append("<b>Fiyat Seviyeleri:</b>")
            if seviyeler.get("hedef_fiyat"):
                lines.append(f"  🎯 Hedef Fiyat: <code>{e(str(seviyeler['hedef_fiyat']))}</code>")
            if seviyeler.get("zarar_kes"):
                lines.append(f"  🛑 Zarar Kes: <code>{e(str(seviyeler['zarar_kes']))}</code>")
            destek = seviyeler.get("destek") or []
            if destek:
                joined = " / ".join(e(str(d)) for d in destek)
                lines.append(f"  🟢 Destek: {joined}")
            direnc = seviyeler.get("direnc") or []
            if direnc:
                joined = " / ".join(e(str(d)) for d in direnc)
                lines.append(f"  🔴 Direnç: {joined}")
            lines.append("")

    oneri = data.get("oneri")
    if oneri:
        emoji = {"Al": "📈", "Sat": "📉", "Tut": "⏸", "Nötr": "➡️"}.get(oneri, "💡")
        lines.append(f"<b>Öneri:</b> {emoji} {e(str(oneri))}")
        lines.append("")

    notlar = data.get("onemli_notlar") or data.get("onemli_gelismeler") or []
    if notlar:
        lines.append("<b>Önemli Notlar:</b>")
        for note in notlar[:5]:
            lines.append(f"  • {e(str(note))}")
        lines.append("")

    kaynaklar = data.get("kaynaklar") or []
    if kaynaklar:
        joined = ", ".join(e(str(k)) for k in kaynaklar[:4])
        lines.append(f"<i>Kaynaklar: {joined}</i>")

    out = "\n".join(lines).strip()
    if not out:
        return (
            "ℹ️ <b>Yanıt üretildi</b> ancak özet alanları boş geldi.\n\n"
            "Model bazen geçerli JSON döndürüp metin alanlarını doldurmayabiliyor. "
            "Aynı soruyu tekrar deneyin veya soruyu biraz değiştirin."
        )
    return out


# ── Command Handlers ───────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        WELCOME_MESSAGE,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def cmd_analiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analiz <TICKER> command."""
    args = context.args
    if not args:
        await update.message.reply_text(
            "Kullanım: /analiz THYAO\n\nVeya aşağıdan bir hisse seçin:",
            reply_markup=stock_quick_access_keyboard(),
        )
        return

    stock_code = args[0].upper().strip()
    await update.message.reply_text(
        f"*{stock_code}* için analiz türü seçin:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=analysis_type_keyboard(stock_code),
    )


async def cmd_haftalik(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /haftalik <TICKER> command."""
    args = context.args
    if not args:
        await update.message.reply_text("Kullanım: /haftalik THYAO")
        return

    stock_code = args[0].upper().strip()
    await _run_weekly_analysis(update, context, stock_code)


async def cmd_kurumlar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show brokerage source selection keyboard."""
    await update.message.reply_text(
        "Hangi kurumun bültenlerini sorgulamak istiyorsunuz?",
        reply_markup=source_selection_keyboard(),
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show LLM model selection keyboard."""
    current = context.user_data.get(_SESSION_MODEL_KEY, _settings.ollama_model)
    await update.message.reply_text(
        f"Aktif model: *{current}*\n\nYeni model seçin:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=model_selection_keyboard(),
    )


async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger ingestion — admin use."""
    await update.message.reply_text("Veri toplama başlatılıyor... Bu birkaç dakika sürebilir.")
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        from app.ingestion.pipeline import run_ingestion_pipeline
        result = await run_ingestion_pipeline()
        await update.message.reply_text(
            f"Veri toplama tamamlandı!\n\n"
            f"Döküman: {result.total_documents}\n"
            f"Chunk: {result.total_chunks}\n"
            f"Eklenen: {result.upserted}\n"
            f"Hata: {len(result.errors)}"
        )
    except Exception as exc:
        logger.error(f"[bot] Manual ingest failed: {exc}")
        await update.message.reply_text(f"Hata oluştu: {exc}")


async def cmd_durum(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show system health status."""
    from app.llm.client import health_check
    from app.vectorstore.client import get_collection_stats

    current_model = context.user_data.get(_SESSION_MODEL_KEY, _settings.ollama_model)
    ollama_ok = await health_check(model=current_model)
    stats = get_collection_stats()

    chroma_addr = f"{stats.get('chroma_host', 'localhost')}:{stats.get('chroma_port', 8001)}"
    status_icon = "🟢" if ollama_ok else "🔴"
    await update.message.reply_text(
        f"*Sistem Durumu*\n\n"
        f"{status_icon} Ollama: {'Bağlı' if ollama_ok else 'Bağlı Değil'}\n"
        f"🤖 Model: {current_model}\n"
        f"📚 Vektör DB ({chroma_addr}): {stats['document_count']} chunk\n"
        f"🗃 Koleksiyon: {stats['collection']}",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Callback Query Handlers ────────────────────────────────────────────────────

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data: str = query.data

    if data.startswith("analiz:"):
        stock = data.split(":", 1)[1]
        if stock == "manual":
            await query.edit_message_text("Analiz için hisse kodunu yazın (ör: THYAO):")
            return
        await query.edit_message_text(
            f"*{stock}* için analiz türü seçin:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=analysis_type_keyboard(stock),
        )

    elif data.startswith("gunluk:"):
        stock = data.split(":", 1)[1]
        await query.edit_message_text(f"*{stock}* günlük analiz yükleniyor...", parse_mode=ParseMode.MARKDOWN)
        await _run_stock_analysis(update, context, stock, weekly=False)

    elif data.startswith("haftalik:"):
        stock = data.split(":", 1)[1]
        await query.edit_message_text(f"*{stock}* haftalık özet yükleniyor...", parse_mode=ParseMode.MARKDOWN)
        await _run_weekly_analysis(update, context, stock)

    elif data.startswith("kaynak:"):
        stock = data.split(":", 1)[1]
        context.user_data["pending_stock"] = stock
        await query.edit_message_text(
            f"*{stock}* için kaynak seçin:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=source_selection_keyboard(),
        )

    elif data.startswith("source:"):
        source_key = data.split(":", 1)[1]
        source = None if source_key == "all" else source_key
        context.user_data[_SESSION_SOURCE_KEY] = source
        source_name = "Tüm Kurumlar" if source_key == "all" else source_key.replace("_", " ").title()
        stock = context.user_data.pop("pending_stock", None)
        if stock:
            await query.edit_message_text(
                f"Kaynak: *{source_name}* — *{stock}* analiz yükleniyor...",
                parse_mode=ParseMode.MARKDOWN,
            )
            await _run_stock_analysis(update, context, stock, source=source, weekly=False)
        else:
            await query.edit_message_text(f"Kaynak *{source_name}* olarak seçildi.", parse_mode=ParseMode.MARKDOWN)

    elif data.startswith("sabah:"):
        preset = (
            "Bugünkü Sabah Stratejisi bülteninde hangi şirket ve kurumlar geçiyor? "
            "Her biri için tek cümle özet ver. Borsa dışı veya kodu olmayan kurumlar için de ayrı madde yaz."
        )
        await query.edit_message_text("📰 Sabah raporu taranıyor…")
        model = context.user_data.get(_SESSION_MODEL_KEY)
        src = context.user_data.get(_SESSION_SOURCE_KEY)
        try:
            from app.rag.chain import free_query

            result = await free_query(query=preset, days_back=14, model=model, source=src)
            formatted = _format_rag_response(result.raw_json)
            await query.message.reply_text(formatted, parse_mode=ParseMode.HTML)
        except Exception as exc:
            logger.error(f"[bot] Sabah preset RAG failed: {exc}", exc_info=True)
            await query.message.reply_text(f"Hata: {exc}")

    elif data.startswith("model:"):
        model_name = data.split(":", 1)[1]
        context.user_data[_SESSION_MODEL_KEY] = model_name
        await query.edit_message_text(f"Model *{model_name}* olarak ayarlandı.", parse_mode=ParseMode.MARKDOWN)


# ── Free-text Message Handler ──────────────────────────────────────────────────

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-form text messages as open RAG queries."""
    text = update.message.text.strip()
    if not text:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    # Check if message looks like a stock code query
    stock_match = re.match(r"^([A-Z]{3,5})\s*\??$", text.upper())
    if stock_match:
        stock_code = stock_match.group(1)
        await update.message.reply_text(
            f"*{stock_code}* hissesi için analiz türü seçin:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=analysis_type_keyboard(stock_code),
        )
        return

    model = context.user_data.get(_SESSION_MODEL_KEY)
    source = context.user_data.get(_SESSION_SOURCE_KEY)

    try:
        from app.rag.chain import free_query

        thinking_msg = await update.message.reply_text("Analiz yapılıyor...")
        result = await free_query(
            query=text,
            days_back=14,
            model=model,
            source=source,
        )
        formatted = _format_rag_response(result.raw_json)

        await thinking_msg.delete()
        await update.message.reply_text(
            formatted,
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        logger.error(f"[bot] Free query failed: {exc}", exc_info=True)
        await update.message.reply_text(f"Bir hata oluştu: {exc}")


# ── Internal Helpers ───────────────────────────────────────────────────────────

async def _run_stock_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    stock_code: str,
    source: Optional[str] = None,
    weekly: bool = False,
) -> None:
    model = context.user_data.get(_SESSION_MODEL_KEY)
    source = source or context.user_data.get(_SESSION_SOURCE_KEY)

    chat = update.effective_chat
    await chat.send_action(ChatAction.TYPING)

    try:
        from app.rag.chain import query_analysis

        query = f"{stock_code} hissesi için teknik analiz, destek direnç seviyeleri ve öneri"
        result = await query_analysis(
            query=query,
            stock_code=stock_code,
            source=source,
            days_back=7,
            model=model,
        )
        formatted = _format_rag_response(result.raw_json)
        await chat.send_message(formatted, parse_mode=ParseMode.HTML)

    except Exception as exc:
        logger.error(f"[bot] Stock analysis failed: {exc}", exc_info=True)
        await chat.send_message(f"Analiz sırasında hata oluştu: {exc}")


async def _run_weekly_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    stock_code: str,
) -> None:
    model = context.user_data.get(_SESSION_MODEL_KEY)
    chat = update.effective_chat
    await chat.send_action(ChatAction.TYPING)

    try:
        from app.rag.chain import query_weekly

        result = await query_weekly(stock_code=stock_code, model=model)
        formatted = _format_rag_response(result.raw_json)
        title = html.escape(stock_code)
        await chat.send_message(
            f"<b>{title} Haftalık Özet</b>\n\n{formatted}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        logger.error(f"[bot] Weekly analysis failed: {exc}", exc_info=True)
        await chat.send_message(f"Haftalık özet sırasında hata oluştu: {exc}")


# ── Application Builder ────────────────────────────────────────────────────────

def build_application() -> Application:
    """Build and configure the Telegram Application instance."""
    app = Application.builder().token(_settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analiz", cmd_analiz))
    app.add_handler(CommandHandler("haftalik", cmd_haftalik))
    app.add_handler(CommandHandler("kurumlar", cmd_kurumlar))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("durum", cmd_durum))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    return app
