"""
LangChain ChatOllama istemcisi.

Ollama, Docker container'da localhost:11434 portunda çalışmaktadır (RTX 5070 GPU).
Bu modül LangChain'i YALNIZCA LLM inference için kullanır.
Embedding ve retrieval işlemleri kendi modüllerinde (SentenceTransformers) yapılır.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger

from app.config import get_settings
from app.llm.prompts import NO_DATA_RESPONSE

_settings = get_settings()


def get_llm(model: Optional[str] = None) -> ChatOllama:
    """
    ChatOllama örneği döndür.

    Args:
        model: Config'deki varsayılanı override etmek için model adı.

    Returns:
        localhost:11434'te çalışan Ollama'ya bağlı ChatOllama nesnesi.
    """
    return ChatOllama(
        base_url=_settings.ollama_base_url,
        model=model or _settings.ollama_model,
        temperature=0.1,
        num_predict=1024,
        top_p=0.9,
    )


async def health_check(model: Optional[str] = None) -> bool:
    """Ollama servisinin erişilebilir ve modelin mevcut olduğunu doğrula."""
    target_model = model or _settings.ollama_model
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            tags_data = resp.json()
            available_models = [m["name"] for m in tags_data.get("models", [])]
            base_model = target_model.split(":")[0]
            is_available = any(base_model in m for m in available_models)
            if not is_available:
                logger.warning(
                    f"[ollama] Model '{target_model}' bulunamadı. "
                    f"Mevcut: {available_models}"
                )
            return is_available
    except Exception as exc:
        logger.error(f"[ollama] Sağlık kontrolü başarısız: {exc}")
        return False


def parse_json_response(raw: str) -> dict:
    """
    LLM çıktısını JSON olarak ayrıştır.
    Markdown kod bloklarını (`json ... `) otomatik olarak soyar.
    """
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    logger.warning("[ollama] JSON ayrıştırılamadı — ham metin döndürülüyor")
    return {"ozet": text, "yeterli_veri": False, "parse_error": True}


async def chat(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """
    LangChain ChatOllama üzerinden sohbet tamamlama.

    Args:
        system_prompt: LLM'e verilen sistem talimatı.
        user_message:  Kullanıcı mesajı (RAG context dahil).
        model:         Override model adı.
        max_tokens:    Maksimum üretilecek token sayısı.

    Returns:
        LLM'nin ürettiği ham metin.
    """
    llm = get_llm(model)
    llm.num_predict = max_tokens

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.debug(f"[ollama] İstek gönderiliyor → model={llm.model}")
    try:
        response = await llm.ainvoke(messages)
        content = response.content
        if not content:
            logger.warning("[ollama] Boş yanıt alındı")
            return NO_DATA_RESPONSE
        logger.debug(f"[ollama] Yanıt alındı — {len(content)} karakter")
        return content
    except Exception as exc:
        logger.error(f"[ollama] LLM çağrısı başarısız: {exc}")
        return NO_DATA_RESPONSE
