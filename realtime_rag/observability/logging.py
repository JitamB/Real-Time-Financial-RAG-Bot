"""Structured JSON logging.

One JSON object per line so logs are greppable and ingestible by any log
pipeline. ``event=...`` is the stable key the README/demo refers to.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        # Attach structured extras (anything passed via logger.info(..., extra={...})).
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent root logging setup. Safe to call from every entrypoint.

    ``level`` controls *our* (``realtime_rag.*``) verbosity. The root logger is
    pinned to WARNING so third-party INFO never reaches the terminal: Pathway's
    ``pathway_engine.persistence.*`` snapshots checkpoint connector offsets to
    ``./Cache`` roughly every second and, with the per-request ``aiohttp.access``
    / root ``http_access`` lines, drown out the events that actually prove
    dynamism. A genuine third-party WARNING/ERROR still propagates (real
    problems aren't hidden); only the INFO-level churn is suppressed. Because
    ``realtime_rag`` carries its own explicit level, its children inherit it
    regardless of the root level.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("realtime_rag").setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event: ``log_event(log, "answer_served", latency_ms=812)``."""
    logger.log(level, event, extra={"event": event, "fields": fields})
