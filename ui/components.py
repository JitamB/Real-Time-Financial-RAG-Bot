"""UI helpers: a thin layer over Pathway's RAGClient + the docs folder.

No SQLite, no CSV, no polling — the old fragile bridge is gone. The UI talks
to the backend over HTTP via the official ``RAGClient``; document add/delete is
just atomic filesystem ops on the watched folder (Channel A).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pathway.xpacks.llm.question_answering import RAGClient

DEFAULT_BACKEND = os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")
DOCS_DIR = Path(os.environ.get("DOCS_DIR", "./data/docs"))
MAX_UPLOAD_BYTES = 8_000_000


def get_client(url: str = DEFAULT_BACKEND) -> RAGClient:
    # Generous timeout: first query may wait on a cold embedder/LLM.
    return RAGClient(url=url, timeout=120)


def backend_stats(client: RAGClient) -> dict | None:
    """`/v1/statistics` doubles as the health/liveness probe."""
    try:
        return client.statistics()
    except Exception:
        return None


def _with_history(prompt: str, history: list[tuple[str, str]] | None) -> str:
    """Prepend a short, clearly-delimited transcript so follow-ups resolve
    ("how does that compare?" needs the prior turn). Bounded by the caller;
    the prefix also seeds retrieval, so keep it terse."""
    if not history:
        return prompt
    lines = [f"{role}: {text}" for role, text in history]
    return (
        "Conversation so far (for context only):\n"
        + "\n".join(lines)
        + f"\n\nCurrent question: {prompt}"
    )


def ask(
    client: RAGClient,
    prompt: str,
    filters: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> dict:
    """Return ``{"response": str, "context_docs": [...]}`` (never raises).

    ``filters``: optional jmespath metadata filter (search-scope control).
    ``history``: optional [(role, text), ...] of prior turns for multi-turn
    follow-ups (backend stays stateless — context is prepended client-side).
    """
    try:
        resp = client.answer(
            _with_history(prompt, history),
            filters=filters,
            return_context_docs=True,
        )
        if isinstance(resp, dict):
            return resp
        return {"response": str(resp), "context_docs": []}
    except Exception as exc:
        return {"response": f"⚠️ Backend error: {exc}", "context_docs": []}


def list_docs(client: RAGClient) -> list[dict]:
    try:
        docs = client.list_documents(keys=["path"]) or []
        return docs if isinstance(docs, list) else []
    except Exception:
        return []


def _safe_name(name: str) -> str:
    return os.path.basename(name).replace(os.sep, "_") or "upload.txt"


def add_document(filename: str, data: bytes, is_pdf: bool) -> tuple[bool, str]:
    """Write a doc into the watched folder *atomically* (temp + os.replace).

    Atomic rename means Pathway's fs watcher never sees a half-written file —
    one clean ``+1`` diff. PDFs are converted to text here (keeps the runtime
    parser lean and avoids poppler).
    """
    if len(data) > MAX_UPLOAD_BYTES:
        return False, f"File too large ({len(data)} bytes > {MAX_UPLOAD_BYTES})."
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_name(filename)
    try:
        if is_pdf or name.lower().endswith(".pdf"):
            import pypdf

            reader = pypdf.PdfReader(__import__("io").BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            # pypdf silently yields empty text for scanned/image-only PDFs.
            # Reject explicitly so the user knows OCR is needed instead of
            # an empty doc being indexed and producing "no information".
            if not text.strip():
                return (
                    False,
                    "PDF appears scanned/image-only — no extractable text "
                    "(needs OCR; set PARSER_BACKEND=docling on the backend).",
                )
            name = os.path.splitext(name)[0] + ".txt"
            payload = text.encode("utf-8")
        else:
            payload = data
        fd, tmp = tempfile.mkstemp(dir=DOCS_DIR, suffix=".part")
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, DOCS_DIR / name)  # atomic
        return True, name
    except Exception as exc:
        return False, f"Could not ingest: {exc}"


def delete_document(name: str) -> tuple[bool, str]:
    """Delete a file from the watched folder (emits a ``-1`` -> chunks removed)."""
    target = DOCS_DIR / _safe_name(name)
    try:
        if target.exists():
            target.unlink()
            return True, name
        return False, "not found"
    except Exception as exc:
        return False, str(exc)


def local_doc_names() -> list[str]:
    if not DOCS_DIR.exists():
        return []
    return sorted(p.name for p in DOCS_DIR.iterdir() if p.is_file() and not p.name.endswith(".part"))
