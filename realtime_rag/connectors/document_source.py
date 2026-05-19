"""Channel A — the watched document folder.

``pw.io.fs.read(..., mode="streaming", with_metadata=True)`` is the primary
demoable dynamism channel: Pathway's filesystem connector natively emits a
``+1`` when a file appears, ``-1``+``+1`` when it is modified, and ``-1`` when
it is deleted. Those diffs flow straight into the DocumentStore — no polling,
no restart. Verified against pathway==0.30.1.
"""

from __future__ import annotations

import pathway as pw

from ..config import Settings
from ..observability.logging import get_logger, log_event

log = get_logger(__name__)


def build_document_source(settings: Settings) -> pw.Table:
    """Streaming filesystem source yielding ``data`` (bytes) + ``_metadata``."""
    path = str(settings.docs_path)
    log_event(log, "fs_source_init", path=path)
    return pw.io.fs.read(
        path,
        format="binary",
        mode="streaming",
        with_metadata=True,
        # small autocommit -> low add/modify/delete -> answer latency
        autocommit_duration_ms=1000,
        name="docs_fs",
    )
