"""Central configuration via pydantic-settings.

All knobs live here with safe defaults; only API keys must be supplied via ``.env``.
Backends are switchable (embedder / LLM / index / parser / RAG mode) without code
changes, which keeps the "hybrid local-default, hosted-optional" promise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedderBackend = Literal["local", "openai", "gemini"]
LLMBackend = Literal["groq", "openai"]
RAGMode = Literal["base", "adaptive"]
IndexBackend = Literal["bruteforce", "usearch"]
ParserBackend = Literal["utf8", "docling"]


class Settings(BaseSettings):
    """Runtime settings. Reads ``.env`` then process env (env wins)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets / external services ---
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    pathway_license_key: str = Field(
        default="demo-license-key-with-telemetry", alias="PATHWAY_LICENSE_KEY"
    )

    # --- Backend selection ---
    embedder_backend: EmbedderBackend = "local"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    llm_backend: LLMBackend = "groq"
    llm_model: str = "groq/llama-3.3-70b-versatile"
    rag_mode: RAGMode = "base"
    index_backend: IndexBackend = "bruteforce"
    # utf8: lightweight, no poppler/OCR — ideal for .txt + UI-converted PDFs.
    # docling: rich PDF/table parsing (heavier at runtime).
    parser_backend: ParserBackend = "utf8"

    # --- Retrieval / chunking ---
    search_topk: int = Field(default=6, ge=1, le=50)
    chunk_max_tokens: int = Field(default=400, ge=64, le=4000)

    # --- Paths ---
    docs_dir: str = "./data/docs"
    cache_dir: str = "./Cache"

    # --- Channel B (live finance/news) cadence ---
    quote_poll_seconds: int = Field(default=20, ge=5, le=3600)
    news_poll_seconds: int = Field(default=300, ge=30, le=86400)

    # --- Serving ---
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # --- Resilience ---
    request_timeout_s: float = Field(default=10.0, ge=1.0, le=120.0)
    llm_capacity: int = Field(default=8, ge=1, le=128)
    llm_max_retries: int = Field(default=6, ge=0, le=20)
    max_file_bytes: int = Field(default=8_000_000, ge=1_000, le=200_000_000)

    @field_validator("docs_dir", "cache_dir")
    @classmethod
    def _expand(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @property
    def docs_path(self) -> Path:
        p = Path(self.docs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def llm_active_key(self) -> str:
        return self.groq_api_key if self.llm_backend == "groq" else self.openai_api_key

    def summary(self) -> dict:
        """Non-secret snapshot for startup logging."""
        return {
            "embedder_backend": self.embedder_backend,
            "embedder_model": self.embedder_model,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "rag_mode": self.rag_mode,
            "index_backend": self.index_backend,
            "parser_backend": self.parser_backend,
            "search_topk": self.search_topk,
            "chunk_max_tokens": self.chunk_max_tokens,
            "docs_dir": self.docs_dir,
            "host": self.host,
            "port": self.port,
            "groq_key_present": bool(self.groq_api_key),
            "newsapi_key_present": bool(self.newsapi_key),
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton so every module sees the same config."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
