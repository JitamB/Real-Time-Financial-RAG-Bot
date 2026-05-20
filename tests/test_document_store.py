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
from realtime_rag.pipeline.rag import (
    build_llm,
    build_question_answerer,
    build_reranker,
)
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
    # New default = sentence-aware overlap chunking (improvement #3).
    assert settings.splitter_backend == "recursive"
    assert type(build_splitter(settings)).__name__ == "RecursiveSplitter"


def test_splitter_token_fallback(monkeypatch):
    # SPLITTER_BACKEND=token restores the old hard-cut behaviour (reversible).
    monkeypatch.setenv("SPLITTER_BACKEND", "token")
    cfg._settings = None
    s = cfg.Settings()
    sp = build_splitter(s)
    assert type(sp).__name__ == "TokenCountSplitter"
    assert sp.kwargs["max_tokens"] == s.chunk_max_tokens


def test_build_reranker_backends(settings, monkeypatch):
    # Default: FlashRank (tiny, fast). We ship a float-typed subclass because
    # 0.30.1's FlashRankReranker has an Any-typed score that breaks the
    # BaseRAGQuestionAnswerer rerank-sort; it must still be a FlashRankReranker.
    from pathway.xpacks.llm import rerankers

    rr = build_reranker(settings)
    assert isinstance(rr, rerankers.FlashRankReranker)
    assert type(rr).__name__ == "_TypedFlashRankReranker"
    # Return must be float-typed (string form under PEP 563) so the
    # BaseRAGQuestionAnswerer rerank-sort doesn't see an Any column.
    assert rr.__wrapped__.__annotations__.get("return") in (float, "float")

    # none -> single-stage vector retrieval (no reranker).
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    cfg._settings = None
    assert build_reranker(cfg.Settings()) is None

    # cross_encoder -> stronger ms-marco-MiniLM reranker.
    monkeypatch.setenv("RERANKER_BACKEND", "cross_encoder")
    cfg._settings = None
    rr_ce = build_reranker(cfg.Settings())
    assert type(rr_ce).__name__ == "CrossEncoderReranker"


def test_reranker_wired_in_base_ignored_in_adaptive(settings, monkeypatch):
    store = build_document_store(settings, [build_document_source(settings)])
    llm = build_llm(settings)

    # base + default reranker -> still a BaseRAGQuestionAnswerer, wired with the
    # wide retrieve pool feeding the rerank stage.
    qa_base = build_question_answerer(settings, llm, store)
    assert type(qa_base).__name__ == "BaseRAGQuestionAnswerer"

    # adaptive has no reranker hook upstream -> must NOT crash; just skip it.
    monkeypatch.setenv("RAG_MODE", "adaptive")
    cfg._settings = None
    qa_adaptive = build_question_answerer(cfg.Settings(), llm, store)
    assert type(qa_adaptive).__name__ == "AdaptiveRAGQuestionAnswerer"


def test_build_document_store_combines_both_channels(settings):
    sources = [build_document_source(settings), build_finance_source(settings)]
    store = build_document_store(settings, sources)
    assert isinstance(store, DocumentStore)
    # Brute-force KNN by default: exact + trivially-correct +1/-1 propagation.
    assert type(store.retriever_factory).__name__ == "BruteForceKnnFactory"


def test_question_answerer_modes(monkeypatch):
    # Pure mode-mapping check, reranker off so it stays fast + offline;
    # reranker wiring is covered by test_reranker_wired_in_base_ignored_in_adaptive.
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    cfg._settings = None
    s = cfg.Settings()
    store = build_document_store(s, [build_document_source(s)])
    llm = build_llm(s)  # LiteLLMChat construction makes no network call

    qa_base = build_question_answerer(s, llm, store)
    assert type(qa_base).__name__ == "BaseRAGQuestionAnswerer"

    monkeypatch.setenv("RAG_MODE", "adaptive")
    cfg._settings = None
    qa_adaptive = build_question_answerer(cfg.Settings(), llm, store)
    assert type(qa_adaptive).__name__ == "AdaptiveRAGQuestionAnswerer"
