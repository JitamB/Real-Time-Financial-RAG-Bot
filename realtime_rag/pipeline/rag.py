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
from pathway.xpacks.llm.document_store import DocumentStore

from ..config import Settings
from ..observability.logging import get_logger, log_event

log = get_logger(__name__)


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


def build_question_answerer(
    settings: Settings, llm: pw.UDF, store: DocumentStore
) -> qa.BaseRAGQuestionAnswerer:
    """``base`` top-k RAG by default; ``adaptive`` widens retrieval iteratively."""
    if settings.rag_mode == "adaptive":
        log_event(log, "qa_init", mode="adaptive")
        return qa.AdaptiveRAGQuestionAnswerer(llm, store)
    log_event(log, "qa_init", mode="base", search_topk=settings.search_topk)
    return qa.BaseRAGQuestionAnswerer(llm, store, search_topk=settings.search_topk)
