"""Config parsing + backend switches (fast, no deps)."""

from __future__ import annotations

import realtime_rag.config as cfg


def test_defaults(settings):
    assert settings.embedder_backend == "local"
    assert settings.llm_backend == "groq"
    assert settings.llm_model == "groq/llama-3.3-70b-versatile"
    assert settings.index_backend == "bruteforce"
    assert settings.parser_backend == "utf8"
    assert settings.rag_mode == "base"
    assert settings.search_topk == 6
    assert settings.summary()["groq_key_present"] is True


def test_quality_profile_defaults(settings):
    # Production/interview profile: overlap chunking + two-stage rerank ON,
    # live-feed dedup ON, bounded multi-turn history ON — all reversible.
    assert settings.splitter_backend == "recursive"
    assert settings.chunk_overlap == 80
    assert settings.reranker_backend == "flashrank"
    assert settings.retrieve_topk == 20
    assert settings.news_dedup is True
    assert settings.chat_history_turns == 3
    s = settings.summary()
    assert s["splitter_backend"] == "recursive"
    assert s["reranker_backend"] == "flashrank"
    assert s["news_dedup"] is True
    assert s["chat_history_turns"] == 3


def test_paths_are_created(settings):
    assert settings.docs_path.is_dir()
    assert settings.cache_path.is_dir()


def test_backend_switches(monkeypatch):
    monkeypatch.setenv("EMBEDDER_BACKEND", "openai")
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("INDEX_BACKEND", "usearch")
    monkeypatch.setenv("RAG_MODE", "adaptive")
    monkeypatch.setenv("PARSER_BACKEND", "docling")
    cfg._settings = None
    s = cfg.Settings()
    assert (s.embedder_backend, s.llm_backend, s.index_backend, s.rag_mode, s.parser_backend) == (
        "openai",
        "openai",
        "usearch",
        "adaptive",
        "docling",
    )


def test_quality_switches_reversible(monkeypatch):
    # Every new knob is reversible to the minimal/old behaviour via one env var.
    monkeypatch.setenv("SPLITTER_BACKEND", "token")
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    monkeypatch.setenv("NEWS_DEDUP", "false")
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "0")
    cfg._settings = None
    s = cfg.Settings()
    assert s.splitter_backend == "token"
    assert s.reranker_backend == "none"
    assert s.news_dedup is False
    assert s.chat_history_turns == 0


def test_invalid_enum_rejected(monkeypatch):
    monkeypatch.setenv("INDEX_BACKEND", "nonsense")
    cfg._settings = None
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cfg.Settings()
