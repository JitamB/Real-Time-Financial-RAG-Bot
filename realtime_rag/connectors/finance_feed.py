"""Channel B — live finance/news as an UPSERT stream.

The old design appended every quote to a file forever, so a query saw dozens of
stale AAPL prices. Here each symbol / article has a stable **primary key**; the
connector runs in ``SessionType.UPSERT`` mode so re-emitting the same key
*replaces* the previous row at the engine level (verified by spike: the stale
row is retracted, not accumulated). That makes a new price a true
*modification* that flows incrementally into the same DocumentStore as Channel A.

``run()`` must never raise — an exception kills the connector thread. All
upstream failures are contained in ``market_clients`` and simply skip a cycle.
"""

from __future__ import annotations

import time

import pathway as pw
from pathway.internals.api import SessionType

from ..config import Settings
from ..observability.logging import get_logger, log_event
from . import market_clients

log = get_logger(__name__)


class FinanceDocSchema(pw.Schema):
    # Stable identity per symbol / article URL -> upsert replaces in place.
    key: str = pw.column_definition(primary_key=True)
    # DocumentStore parses `data` (bytes) and filters/attributes on `_metadata`.
    data: bytes
    _metadata: pw.Json


class FinanceFeedSubject(pw.io.python.ConnectorSubject):
    """Polls yfinance + NewsAPI and upserts one row per symbol / article."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._stop = False
        # key -> last emitted content signature. Lets us skip re-emitting a
        # row whose material content is unchanged (NewsAPI returns the same
        # top articles every poll; quotes rarely move between polls), avoiding
        # redundant embed work in the DocumentStore.
        self._last_sig: dict[str, str] = {}

    @property
    def _session_type(self) -> SessionType:
        # Same primary key re-emitted -> engine replaces the row (upsert).
        return SessionType.UPSERT

    def on_stop(self) -> None:  # graceful shutdown hook
        self._stop = True

    def _emit(self, key: str, text: str, kind: str, signature: str) -> bool:
        """Upsert a row unless its content signature is unchanged.

        Returns True if a row was emitted, False if skipped (dedup). The
        ``ts``-only delta never changes the signature, so an unchanged quote/
        article is a true no-op instead of a redundant re-embed.
        """
        if self._settings.news_dedup and self._last_sig.get(key) == signature:
            return False
        self.next(
            key=key,
            data=text.encode("utf-8"),
            # `path` is the metadata key BaseRAGQuestionAnswerer attributes
            # context on; keep it human-readable for the demo.
            _metadata={"path": key, "kind": kind, "source": "finance_feed"},
        )
        self._last_sig[key] = signature
        return True

    def run(self) -> None:
        s = self._settings
        log_event(
            log,
            "finance_feed_start",
            quote_poll_s=s.quote_poll_seconds,
            news_poll_s=s.news_poll_seconds,
            newsapi=bool(s.newsapi_key),
        )
        last_news = 0.0
        while not self._stop:
            try:
                emitted = skipped = 0
                for q in market_clients.fetch_quotes(timeout=s.request_timeout_s):
                    if self._emit(q.key, q.as_document(), "quote", q.content_signature):
                        emitted += 1
                    else:
                        skipped += 1

                now = time.monotonic()
                if now - last_news >= s.news_poll_seconds:
                    for a in market_clients.fetch_news(
                        s.newsapi_key, timeout=s.request_timeout_s
                    ):
                        if self._emit(a.key, a.as_document(), "news", a.content_signature):
                            emitted += 1
                        else:
                            skipped += 1
                    last_news = now

                if emitted or skipped:
                    log_event(
                        log, "finance_feed_cycle", emitted=emitted, deduped=skipped
                    )
                # Push this batch to the engine so the index updates promptly.
                self.commit()
            except Exception as exc:  # never let the connector thread die
                log_event(log, "finance_feed_cycle_error", level=40, error=str(exc))

            # Sleep in short slices so on_stop() is honoured quickly.
            slept = 0
            while slept < s.quote_poll_seconds and not self._stop:
                time.sleep(1)
                slept += 1


def build_finance_source(settings: Settings) -> pw.Table:
    """Streaming UPSERT table; shares the DocumentStore with Channel A."""
    return pw.io.python.read(
        FinanceFeedSubject(settings),
        schema=FinanceDocSchema,
        autocommit_duration_ms=1000,
        name="finance_feed",
    )
