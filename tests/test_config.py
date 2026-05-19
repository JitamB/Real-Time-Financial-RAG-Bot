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


def test_invalid_enum_rejected(monkeypatch):
    monkeypatch.setenv("INDEX_BACKEND", "nonsense")
    cfg._settings = None
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cfg.Settings()
