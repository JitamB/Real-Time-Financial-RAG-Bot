"""Wire the whole live RAG and serve it over REST.

This replaces the entire fragile ``CSV -> pandas -> SQLite -> poll`` bridge of
the old design with Pathway's native REST connector. Endpoints (from
``QASummaryRestServer``, verified against pathway==0.30.1):

    POST /v2/answer          {prompt, filters?, model?, return_context_docs?}
    POST /v2/summarize       {...}
    POST /v2/list_documents  {filters?, keys?}
    POST /v1/retrieve        {query, k, metadata_filter?, filepath_globpattern?}
    GET  /v1/statistics      -> {file_count,last_modified,last_indexed}  (health)

``QASummaryRestServer.run()`` builds the persistence/UDF cache and calls
``pw.run`` itself; extra kwargs (``terminate_on_error``) are forwarded there.
"""

from __future__ import annotations

import pathway as pw
from pathway.xpacks.llm.servers import QASummaryRestServer

from ..config import Settings, get_settings
from ..connectors.document_source import build_document_source
from ..connectors.finance_feed import build_finance_source
from ..observability.logging import configure_logging, get_logger, log_event
from ..pipeline.document_store import build_document_store
from ..pipeline.rag import build_llm, build_question_answerer

log = get_logger(__name__)


def build_app(settings: Settings | None = None) -> QASummaryRestServer:
    """Construct the full pipeline and return an unstarted REST server.

    Kept separate from :func:`run` so tests can build the graph without
    binding a port or calling ``pw.run``.
    """
    settings = settings or get_settings()
    log_event(log, "boot", **settings.summary())

    # Two streaming sources feed ONE incremental DocumentStore.
    sources = [build_document_source(settings)]  # Channel A (always on)
    try:
        sources.append(build_finance_source(settings))  # Channel B (best-effort)
    except Exception as exc:  # never let Channel B block startup
        log_event(log, "finance_source_disabled", level=40, error=str(exc))

    store = build_document_store(settings, sources)
    llm = build_llm(settings)
    answerer = build_question_answerer(settings, llm, store)

    server = QASummaryRestServer(settings.host, settings.port, answerer)
    log_event(log, "server_built", host=settings.host, port=settings.port)
    return server


def run(settings: Settings | None = None) -> None:
    """Boot the live RAG backend (blocks on ``pw.run``)."""
    configure_logging()
    settings = settings or get_settings()
    if settings.pathway_license_key:
        try:
            pw.set_license_key(settings.pathway_license_key)
        except Exception as exc:  # license optional for Community features
            log_event(log, "license_skip", level=30, error=str(exc))

    server = build_app(settings)
    log_event(log, "server_run", url=f"http://{settings.host}:{settings.port}")
    # run() internally: builds pw.persistence.Config(UDF_CACHING) from
    # cache_backend, sets monitoring_level=NONE, then pw.run(**kwargs).
    server.run(
        with_cache=True,
        cache_backend=pw.persistence.Backend.filesystem(str(settings.cache_path)),
        terminate_on_error=False,  # one bad file/LLM call must not kill the engine
    )
