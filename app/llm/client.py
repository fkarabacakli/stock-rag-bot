"""
LangChain ChatOllama istemcisi.

Ollama, Docker container'da localhost:11434 portunda çalişmaktadir (RTX 5070 GPU).
Bu modül LangChain'i YALNIZCA LLM inference için kullanir.
Embedding ve retrieval işlemleri kendi modüllerinde (SentenceTransformers) yapilir.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx
from langchain_ollama import ChatOllama
from loguru import logger

from app.config import get_settings

_settings = get_settings()


def get_llm(model: Optional[str] = None, num_predict: int = 1024) -> ChatOllama:
    """
    Return a ChatOllama instance.

    Args:
        model:       Model name to override the config default.
        num_predict: Maximum tokens to generate. Use a higher value for
                     multi-company overview responses.

    Returns:
        ChatOllama object connected to Ollama running on localhost:11434.
    """
    return ChatOllama(
        base_url=_settings.ollama_base_url,
        model=model or _settings.ollama_model,
        temperature=0.1,
        num_predict=num_predict,
        top_p=0.9,
        format="json",
    )


async def health_check(model: Optional[str] = None) -> bool:
    """Validate that Ollama is reachable and the model is available."""
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
                    f"[ollama] Model '{target_model}' not found. "
                    f"Available: {available_models}"
                )
            return is_available
    except Exception as exc:
        logger.error(f"[ollama] Health check failed: {exc}")
        return False


def parse_json_response(raw: str) -> dict:
    """
    Parse LLM output as JSON.
    Automatically strips Markdown code fences (`json ...`).
    """
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # strip opening fence (```json etc.)
        try:
            close = next(i for i, l in enumerate(lines) if l.strip() == "```")
            lines = lines[:close]
        except StopIteration:
            pass
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    logger.warning("[ollama] Could not parse JSON — returning raw text")
    return {"ozet": text, "yeterli_veri": False, "parse_error": True}


