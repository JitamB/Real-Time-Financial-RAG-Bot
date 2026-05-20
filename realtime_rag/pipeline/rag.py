"""LLM + question-answerer factories.

The old design called Groq **synchronously inside a pw.udf**, stalling the
dataflow worker 1-3s per query. Here the LLM is an async Pathway UDF
(``async_mode="fully_async"``) with bounded concurrency, exponential-backoff
retries, and content-keyed caching — so a slow/failed LLM degrades gracefully
instead of blocking or crashing the engine.

Verified against pathway==0.30.1.
"""

from __future__ import annotations

import pathway as pw
from pathway.xpacks.llm import llms
from pathway.xpacks.llm import question_answering as qa
from pathway.xpacks.llm import rerankers
from pathway.xpacks.llm.document_store import DocumentStore

from ..config import Settings
from ..observability.logging import get_logger, log_event

log = get_logger(__name__)


class _TypedFlashRankReranker(rerankers.FlashRankReranker):
    """FlashRank reranker with an explicit ``float`` return annotation.

    Upstream bug (pathway 0.30.1): ``rerankers.FlashRankReranker.__wrapped__``
    has *no* return annotation, so its UDF column dtype resolves to ``Any``.
    ``BaseRAGQuestionAnswerer`` then builds ``sort_key=-reranker_score`` and the
    engine raises ``TypeError: ... unary operator neg on column of type Any``,
    making FlashRank unusable as a reranker out of the box (CrossEncoder /
    Encoder rerankers annotate ``-> float`` and work). Re-declaring
    ``__wrapped__`` with ``-> float`` fixes the dtype with no behaviour change.
    """

    def __wrapped__(self, doc: str, query: str) -> float:
        return float(super().__wrapped__(doc, query))


def build_llm(settings: Settings) -> pw.UDF:
    """Groq (via LiteLLM ``groq/`` prefix) by default; OpenAI optional."""
    common = dict(
        capacity=settings.llm_capacity,
        retry_strategy=pw.udfs.ExponentialBackoffRetryStrategy(
            max_retries=settings.llm_max_retries
        ),
        cache_strategy=pw.udfs.DefaultCache(),
        async_mode="fully_async",  # never block the dataflow worker
        temperature=0,
    )
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            log_event(log, "llm_key_missing", level=30, backend="openai")
        log_event(log, "llm_init", backend="openai", model=settings.llm_model)
        return llms.OpenAIChat(
            model=settings.llm_model, api_key=settings.openai_api_key, **common
        )

    # Groq through LiteLLM. Model id must carry the provider prefix, e.g.
    # "groq/llama-3.3-70b-versatile".
    if not settings.groq_api_key:
        log_event(log, "llm_key_missing", level=30, backend="groq")
    log_event(log, "llm_init", backend="groq", model=settings.llm_model)
    return llms.LiteLLMChat(
        model=settings.llm_model, api_key=settings.groq_api_key, **common
    )


def build_reranker(settings: Settings) -> pw.UDF | None:
    """Optional second-stage reranker (vector recall -> reorder by relevance).

    ``flashrank`` (default): tiny TinyBERT cross-encoder, ~ms — quality with a
    negligible latency hit. ``cross_encoder``: ms-marco-MiniLM, stronger but
    heavier. ``none``: vector-similarity order only.

    Construction failure (missing optional dep, signature drift) degrades to
    ``None`` with a warning — never breaks the dataflow graph.
    """
    backend = settings.reranker_backend
    if backend == "none":
        return None
    try:
        if backend == "flashrank":
            # _TypedFlashRankReranker, not the upstream class: 0.30.1's
            # FlashRankReranker yields an Any-typed score that breaks the
            # BaseRAGQuestionAnswerer sort (see the subclass docstring).
            rr = _TypedFlashRankReranker(model_name=settings.flashrank_model)
            log_event(log, "reranker_init", backend="flashrank", model=settings.flashrank_model)
            return rr
        if backend == "cross_encoder":
            rr = rerankers.CrossEncoderReranker(settings.cross_encoder_model)
            log_event(
                log, "reranker_init", backend="cross_encoder", model=settings.cross_encoder_model
            )
            return rr
    except Exception as exc:
        log_event(log, "reranker_fallback", level=30, requested=backend, error=str(exc))
    return None


def build_question_answerer(
    settings: Settings, llm: pw.UDF, store: DocumentStore
) -> qa.BaseRAGQuestionAnswerer:
    """``base`` top-k RAG (default) with optional two-stage reranking;
    ``adaptive`` widens retrieval iteratively (no reranker support upstream).
    """
    reranker = build_reranker(settings)

    if settings.rag_mode == "adaptive":
        if reranker is not None:
            # AdaptiveRAGQuestionAnswerer has no reranker arg in 0.30.1 —
            # honour the mode choice, just skip reranking (documented limit).
            log_event(log, "reranker_ignored_adaptive", level=30)
        log_event(log, "qa_init", mode="adaptive")
        return qa.AdaptiveRAGQuestionAnswerer(llm, store)

    if reranker is not None:
        # Retrieve a wide pool by vector similarity, then the reranker narrows
        # it back down to search_topk by cross-encoder relevance.
        log_event(
            log,
            "qa_init",
            mode="base",
            reranker=settings.reranker_backend,
            retrieve_topk=settings.retrieve_topk,
            rerank_topk=settings.search_topk,
        )
        return qa.BaseRAGQuestionAnswerer(
            llm,
            store,
            search_topk=settings.retrieve_topk,
            reranker=reranker,
            rerank_topk=settings.search_topk,
        )

    log_event(log, "qa_init", mode="base", search_topk=settings.search_topk)
    return qa.BaseRAGQuestionAnswerer(llm, store, search_topk=settings.search_topk)
