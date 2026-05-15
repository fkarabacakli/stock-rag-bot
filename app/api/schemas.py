from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Natural language question")
    stock_code: Optional[str] = Field(None, description="BIST ticker filter (e.g. THYAO)")
    source: Optional[str] = Field(None, description="Source brokerage filter")
    days_back: int = Field(7, ge=1, le=90, description="How many days back to search")
    model: Optional[str] = Field(None, description="Ollama model override")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "THYAO hissesi için bu haftaki destek ve direnç seviyeleri neler?",
            "stock_code": "THYAO",
            "days_back": 7,
        }
    }}


class WeeklyQueryRequest(BaseModel):
    stock_code: str = Field(..., min_length=2, max_length=10, description="BIST ticker")
    model: Optional[str] = Field(None, description="Ollama model override")


class FreeQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    days_back: int = Field(14, ge=1, le=90)
    model: Optional[str] = None
    source: Optional[str] = Field(None, description="Source brokerage filter")


# ── Response Models ──────────────────────────────────────────────────────────

class SourceInfo(BaseModel):
    source: str
    date: str
    url: str
    stock_code: str


class QueryResponse(BaseModel):
    success: bool
    query: str
    result: dict
    sources: list[SourceInfo]
    model_used: str
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    ollama_model: str
    chroma_document_count: int
    chroma_collection: str
    chroma_host: str
    chroma_port: int


class StatsResponse(BaseModel):
    collection: str
    chroma_host: str
    chroma_port: int
    document_count: int


class IngestTriggerResponse(BaseModel):
    success: bool
    message: str
    total_documents: int = 0
    total_chunks: int = 0
    upserted: int = 0
    errors: list[str] = []


class CollectionResetResponse(BaseModel):
    success: bool
    message: str
    previous_count: int
