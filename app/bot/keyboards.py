"""
Telegram inline keyboard builders.
"""
from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Brokerage source options
SOURCES = {
    "ziraat_yatirim": "Ziraat Yatirim",
    "halk_yatirim": "Halk Yatirim",
    "all": "Tüm Kurumlar",
}

# Commonly queried BIST stocks for quick access
POPULAR_STOCKS = [
    ("THYAO", "THYAO"), ("GARAN", "GARAN"),
    ("SISE", "SISE"), ("ASELS", "ASELS"),
    ("EREGL", "EREGL"), ("AKBNK", "AKBNK"),
    ("TUPRS", "TUPRS"), ("KCHOL", "KCHOL"),
]


def source_selection_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for selecting a brokerage source."""
    buttons = [
        [InlineKeyboardButton(label, callback_data=f"source:{key}")]
        for key, label in SOURCES.items()
    ]
    return InlineKeyboardMarkup(buttons)


def main_menu_keyboard() -> InlineKeyboardMarkup:
    """Quick stocks plus a preset for the morning report."""
    base = stock_quick_access_keyboard()
    rows = list(base.inline_keyboard)
    rows.append(
        [
            InlineKeyboardButton(
                "📰 Bugün hangi şirketler?",
                callback_data="sabah:sirketler",
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


def stock_quick_access_keyboard() -> InlineKeyboardMarkup:
    """Quick-access keyboard for popular BIST stocks."""
    rows = []
    row = []
    for ticker, label in POPULAR_STOCKS:
        row.append(InlineKeyboardButton(label, callback_data=f"analiz:{ticker}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("Manuel Giriş", callback_data="analiz:manual")])
    return InlineKeyboardMarkup(rows)


def analysis_type_keyboard(stock_code: str) -> InlineKeyboardMarkup:
    """Choose between single-day analysis and weekly summary."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Günlük Analiz", callback_data=f"gunluk:{stock_code}"
            ),
            InlineKeyboardButton(
                "Haftalik Özet", callback_data=f"haftalik:{stock_code}"
            ),
        ],
        [InlineKeyboardButton("Kaynak Seç", callback_data=f"kaynak:{stock_code}")],
    ])


def model_selection_keyboard() -> InlineKeyboardMarkup:
    """Allow user to switch the LLM model for the session."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Qwen 2.5 7B", callback_data="model:qwen2.5:7b"),
            InlineKeyboardButton("Llama 3.1 8B", callback_data="model:llama3.1:8b"),
        ]
    ])
