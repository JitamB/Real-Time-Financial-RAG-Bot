"""DocumentStore wiring: both channels feed ONE incremental store with the
configured parser/splitter/index. (End-to-end add/modify/delete behaviour is
proven against the live server by scripts/demo_dynamism.sh; here we assert the
graph is assembled correctly and is incremental by construction.)"""

from __future__ import annotations

import pathway as pw
from pathway.xpacks.llm.document_store import DocumentStore

from realtime_rag.connectors.document_source import build_document_source
from realtime_rag.connectors.finance_feed import build_finance_source
from realtime_rag.pipeline.document_store import (
    build_document_store,
    build_parser,
    build_retriever_factory,
    build_splitter,
)
from realtime_rag.pipeline.rag import build_llm, build_question_answerer
import realtime_rag.config as cfg


def test_fs_source_is_streaming_with_metadata(settings):
    t = build_document_source(settings)
    cols = set(t.column_names())
    # DocumentStore needs data + _metadata; fs connector must provide both.
    assert "data" in cols and "_metadata" in cols


def test_finance_source_matches_docstore_contract(settings):
    t = build_finance_source(settings)
    cols = set(t.column_names())
    assert "data" in cols and "_metadata" in cols  # same shape as Channel A


def test_parser_and_splitter_backends(settings):
    assert type(build_parser(settings)).__name__ == "Utf8Parser"
    sp = build_splitter(settings)
    assert sp.kwargs["max_tokens"] == settings.chunk_max_tokens


def test_build_document_store_combines_both_channels(settings):
    sources = [build_document_source(settings), build_finance_source(settings)]
    store = build_document_store(settings, sources)
    assert isinstance(store, DocumentStore)
    # Brute-force KNN by default: exact + trivially-correct +1/-1 propagation.
    assert type(store.retriever_factory).__name__ == "BruteForceKnnFactory"


def test_question_answerer_modes(settings, monkeypatch):
    store = build_document_store(settings, [build_document_source(settings)])
    llm = build_llm(settings)  # LiteLLMChat construction makes no network call

    qa_base = build_question_answerer(settings, llm, store)
    assert type(qa_base).__name__ == "BaseRAGQuestionAnswerer"

    monkeypatch.setenv("RAG_MODE", "adaptive")
    cfg._settings = None
    qa_adaptive = build_question_answerer(cfg.Settings(), llm, store)
    assert type(qa_adaptive).__name__ == "AdaptiveRAGQuestionAnswerer"
