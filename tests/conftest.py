"""Shared fixtures.

Tests are deterministic and offline: a tiny *cached* local embedder, finite
Pathway subjects run to completion in static/batch mode, no network, no LLM.
Pathway's batch == streaming semantics, so static-mode assertions about which
documents are retrievable also prove the add/delete behaviour.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Small, fast, already in the HF cache on dev machines / CI base image.
TEST_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """Fresh Settings pointed at a temp docs/cache dir with the tiny embedder."""
    monkeypatch.setenv("EMBEDDER_MODEL", TEST_EMBED_MODEL)
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    monkeypatch.setenv("NEWSAPI_KEY", "")
    # Defeat the get_settings() singleton between tests.
    import realtime_rag.config as cfg

    cfg._settings = None
    s = cfg.Settings()
    return s


@pytest.fixture()
def docs_dir(settings):
    d = settings.docs_path
    return d
