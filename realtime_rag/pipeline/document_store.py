"""Incremental ``DocumentStore`` factory — the heart of the dynamism.

The store is a *standing* Pathway computation over the source tables:
``parse -> split -> embed -> KNN index``. Because Pathway is a differential
dataflow engine, a ``+1``/``-1`` diff from any source (a file added/edited/
deleted, or a finance row upserted) propagates here automatically and only the
affected chunks are re-embedded / removed from the index. No restart, no batch
re-indexing — see ARCHITECTURE.md.

API verified against the installed pathway==0.30.1 (signatures differ from older
releases; do not "fix" these to match outdated docs without re-introspecting).
"""

from __future__ import annotations

from collections.abc import Iterable

import pathway as pw
from pathway.xpacks.llm import embedders, parsers, splitters
from pathway.xpacks.llm.document_store import DocumentStore

from ..config import Settings
from ..observability.logging import get_logger, log_event

log = get_logger(__name__)


def build_embedder(settings: Settings) -> pw.UDF:
    """Hybrid embedder: local sentence-transformers by default, hosted optional.

    The local path needs no API key or network — the most reliable profile for
    a live demo. Hosted paths get retries + content-keyed caching for free.
    """
    backend = settings.embedder_backend
    if backend == "local":
        log_event(log, "embedder_init", backend="local", model=settings.embedder_model)
        # NOTE: SentenceTransformerEmbedder has no retry/cache kwargs (it is a
        # local, synchronous model) — passing them would raise.
        return embedders.SentenceTransformerEmbedder(
            model=settings.embedder_model,
            call_kwargs={"show_progress_bar": False},
        )
    if backend == "openai":
        log_event(log, "embedder_init", backend="openai", model=settings.embedder_model)
        return embedders.OpenAIEmbedder(
            model=settings.embedder_model,
            api_key=settings.openai_api_key,
            cache_strategy=pw.udfs.DefaultCache(),
        )
    if backend == "gemini":
        log_event(log, "embedder_init", backend="gemini", model=settings.embedder_model)
        return embedders.GeminiEmbedder(
            model=settings.embedder_model,
            api_key=settings.gemini_api_key,
            cache_strategy=pw.udfs.DefaultCache(),
        )
    raise ValueError(f"unknown embedder_backend: {backend}")


def build_parser(settings: Settings) -> pw.UDF:
    """``utf8`` (lean, no poppler/OCR) by default; ``docling`` for rich PDF.

    Our Channel A docs are text, and the UI converts uploaded PDFs to text
    before dropping them in, so utf8 is correct and dependency-light at runtime.
    """
    if settings.parser_backend == "docling":
        log_event(log, "parser_init", backend="docling")
        # chunk=False: let our TokenCountSplitter own chunking uniformly across
        # both channels (consistent chunk sizes -> consistent retrieval).
        return parsers.DoclingParser(chunk=False, async_mode="fully_async")
    log_event(log, "parser_init", backend="utf8")
    return parsers.Utf8Parser()


def build_splitter(settings: Settings) -> pw.UDF:
    log_event(log, "splitter_init", max_tokens=settings.chunk_max_tokens)
    return splitters.TokenCountSplitter(max_tokens=settings.chunk_max_tokens)


def build_retriever_factory(settings: Settings, embedder: pw.UDF):
    """Incremental KNN factory.

    Default ``bruteforce``: exact, no capacity tuning, +1/-1 propagation is
    trivially correct — ideal for a demo corpus. ``usearch`` (HNSW) is the
    sub-linear option for the "path to production" story.
    """
    dim = embedder.get_embedding_dimension()
    if settings.index_backend == "usearch":
        log_event(log, "index_init", backend="usearch", dim=dim)
        return pw.indexing.UsearchKnnFactory(embedder=embedder, dimensions=dim)
    log_event(log, "index_init", backend="bruteforce", dim=dim)
    return pw.indexing.BruteForceKnnFactory(embedder=embedder, dimensions=dim)


def build_document_store(
    settings: Settings, sources: pw.Table | Iterable[pw.Table]
) -> DocumentStore:
    """Assemble the incremental vector store over ``sources``.

    Each source table must expose a ``data`` (bytes) column and a ``_metadata``
    column (DocumentStore filters/identifies documents on it). The filesystem
    connector provides both via ``with_metadata=True``; the finance connector
    emits them explicitly (see connectors/finance_feed.py).
    """
    embedder = build_embedder(settings)
    store = DocumentStore(
        docs=sources,
        retriever_factory=build_retriever_factory(settings, embedder),
        parser=build_parser(settings),
        splitter=build_splitter(settings),
    )
    log_event(log, "document_store_built")
    return store
