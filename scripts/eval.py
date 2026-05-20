#!/usr/bin/env python3
"""Slim answer-quality eval against a running backend.

Deliberately *not* RAGAS: that pulls langchain/datasets and needs a networked
judge LLM — heavy and flaky for a demo. This is a dependency-free harness
(stdlib + the RAGClient we already ship) that measures grounded recall over a
small committed golden set, with an optional LLM faithfulness judge that reuses
the project's own Groq model.

It is **not** part of the default ``pytest`` run — it needs a live backend.

    # 1. start the backend
    ./venv/bin/python -m realtime_rag.app
    # 2. (optional) seed the grounded fixture into the watched folder
    cp data/docs/seed_market_glossary.txt "$DOCS_DIR"/   # if not already there
    # 3. run the eval
    ./venv/bin/python scripts/eval.py                # keyword recall only
    ./venv/bin/python scripts/eval.py --judge        # + LLM faithfulness 0-1

Exit code is 0 unless ``--fail-under R`` is given and mean recall < R (lets it
double as a CI gate later without changing the default behaviour).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GOLDEN = _REPO_ROOT / "tests" / "golden" / "seed_glossary_qa.json"
DEFAULT_BACKEND = os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")

_JUDGE_SYSTEM = (
    "You are a strict grading assistant. Given a QUESTION, the retrieved "
    "CONTEXT, and an ANSWER, rate ONLY how well the ANSWER is supported by the "
    "CONTEXT (faithfulness/grounding), ignoring style. Reply with a single "
    "number between 0 and 1 (e.g. 0.0, 0.5, 1.0) and nothing else."
)


def _doc_text(doc: object) -> str:
    """Best-effort text extraction from a context_docs entry."""
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("data") or doc)
    return str(doc)


def keyword_recall(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Fraction of expected keywords present (case-insensitive substring)."""
    if not keywords:
        return 1.0, []
    low = answer.lower()
    missing = [k for k in keywords if k.lower() not in low]
    return (len(keywords) - len(missing)) / len(keywords), missing


def judge_faithfulness(question: str, context: str, answer: str) -> float | None:
    """0-1 grounding score from the project's own Groq LLM (via litellm).

    Returns ``None`` if the judge is unavailable (no key / API error) so the
    recall report still prints.
    """
    try:
        import litellm

        from realtime_rag.config import get_settings

        s = get_settings()
        resp = litellm.completion(
            model=s.llm_model,
            api_key=s.llm_active_key,
            temperature=0,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"CONTEXT:\n{context[:6000]}\n\n"
                        f"ANSWER:\n{answer}\n\nScore:"
                    ),
                },
            ],
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        # Pull the first float-looking token; clamp to [0, 1].
        tok = raw.split()[0].strip().rstrip(".")
        return max(0.0, min(1.0, float(tok)))
    except Exception as exc:  # never let the judge break the recall report
        print(f"  (judge unavailable: {exc})", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Slim grounded-recall eval.")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, help="RAG backend URL")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="golden JSON path")
    ap.add_argument("--judge", action="store_true", help="also LLM-judge faithfulness")
    ap.add_argument("--timeout", type=int, default=120, help="per-query timeout (s)")
    ap.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="R",
        help="exit non-zero if mean recall < R (default: never fail)",
    )
    args = ap.parse_args()

    from pathway.xpacks.llm.question_answering import RAGClient

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pairs = golden.get("pairs", [])
    grounded_in = golden.get("grounded_in", "?")
    if not pairs:
        print("no pairs in golden set", file=sys.stderr)
        return 2

    client = RAGClient(url=args.backend, timeout=args.timeout)
    print(f"Backend : {args.backend}")
    print(f"Golden  : {args.golden}  ({len(pairs)} pairs, grounded in {grounded_in})")
    print(f"Judge   : {'on' if args.judge else 'off'}")
    print("-" * 72)

    recalls: list[float] = []
    judged: list[float] = []
    passes = 0

    for i, pair in enumerate(pairs, 1):
        q = pair["question"]
        kws = pair.get("keywords", [])
        try:
            resp = client.answer(q, return_context_docs=True)
        except Exception as exc:
            print(f"[{i:>2}] ERROR querying backend: {exc}")
            recalls.append(0.0)
            continue

        if isinstance(resp, dict):
            answer = str(resp.get("response", ""))
            ctx_docs = resp.get("context_docs") or []
        else:
            answer, ctx_docs = str(resp), []

        recall, missing = keyword_recall(answer, kws)
        recalls.append(recall)
        ok = recall == 1.0
        passes += int(ok)

        flag = "PASS" if ok else "FAIL"
        line = f"[{i:>2}] {flag}  recall={recall:0.2f}"
        if args.judge:
            ctx = "\n---\n".join(_doc_text(d) for d in ctx_docs)
            score = judge_faithfulness(q, ctx, answer)
            if score is not None:
                judged.append(score)
                line += f"  faithful={score:0.2f}"
        print(line + f"  | {q}")
        if missing:
            print(f"       missing keywords: {missing}")

    n = len(pairs)
    mean_recall = sum(recalls) / n if n else 0.0
    print("-" * 72)
    print(f"Pass@all-keywords : {passes}/{n}  ({passes / n:.0%})")
    print(f"Mean recall       : {mean_recall:0.3f}")
    if judged:
        print(f"Mean faithfulness : {sum(judged) / len(judged):0.3f}  (n={len(judged)})")

    if args.fail_under is not None and mean_recall < args.fail_under:
        print(f"FAIL: mean recall {mean_recall:0.3f} < --fail-under {args.fail_under}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
