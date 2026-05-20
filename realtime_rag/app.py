"""Entrypoint: ``python -m realtime_rag.app`` (or the ``realtime-rag`` script).

Thin by design — all wiring lives in server/rest_app.py and is driven by
config.py (the single source of truth, env-overridable). This mirrors the
official Pathway llm-app template structure.
"""

from __future__ import annotations

from .server.rest_app import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
