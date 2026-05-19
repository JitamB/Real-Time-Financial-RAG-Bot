# Real-Time Incremental Financial RAG — powered by Pathway

A Retrieval-Augmented Generation system that **thinks in real time**. Add, modify,
or delete a document in the watched folder — or let the live market/news feed
update — and the chatbot's answer to a relevant question changes within
**seconds**, with **no manual restart and no batch re-indexing**.

Built on [Pathway](https://pathway.com)'s differential-dataflow engine and its
native incremental `DocumentStore`. This is the property the
["Dynamic RAG Playground"](PS/ps.txt) challenge scores highest (35%).

---

## Why this is different

| | Typical RAG | This project |
|---|---|---|
| New document | re-run an indexing batch job | indexed incrementally in seconds, automatically |
| Edited document | stale until next batch | old chunks retracted, new chunks indexed, automatically |
| **Deleted document** | usually **never** removed | chunks removed from the vector index, automatically |
| Live market data | appended forever (grows unbounded) | **upserted** by key — only the latest state is retrieved |
| Stale answers | LRU cache can serve outdated replies | impossible — caching is content-keyed |

See [ARCHITECTURE.md](ARCHITECTURE.md) for *why* deletion "just works" (differential
dataflow), the data-flow diagram, and the resilience/scaling notes.

## Architecture at a glance

```
Channel A: data/docs/  ──(fs streaming, +1/-1/modify)──┐
                                                        ├─▶ Pathway DocumentStore
Channel B: yfinance+NewsAPI ──(Python UPSERT connector)─┘   parse→split→embed→KNN
                                                                     │
HTTP  POST /v2/answer ──▶ rest_connector ──▶ retrieve top-k ──▶ async LiteLLM (Groq)
Streamlit UI ──(pathway RAGClient over HTTP; no SQLite/CSV/polling)──┘
```

- **Engine**: Pathway 0.30.x (`xpacks.llm` `DocumentStore`, incremental KNN).
- **Embeddings**: local `BAAI/bge-small-en-v1.5` by default (offline, no quota);
  switchable to OpenAI/Gemini via config.
- **LLM**: Groq `llama-3.3-70b-versatile` via LiteLLM, called as a non-blocking
  async UDF with retries.
- **Serving**: Pathway REST API; Streamlit dashboard talks to it directly.

## Quick start

### Option A — Docker (recommended)

```bash
cp .env.example .env          # then put your GROQ_API_KEY (and optional NEWSAPI_KEY) in .env
docker compose up --build     # backend :8000, UI :8501
```

Open <http://localhost:8501> for the dashboard, or query the API directly:

```bash
curl -fsS http://localhost:8000/v1/statistics            # health
scripts/smoke_query.sh "What is in the seed company brief?"
```

### Option B — local Python (3.11)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY

# Terminal 1 — backend (REST :8000)
python -m realtime_rag.app

# Terminal 2 — Streamlit UI (:8501)
streamlit run ui/ui.py
```

> The `pathway[xpack-llm]` *extra* is intentionally **not** used — it makes pip
> fail with `resolution-too-deep`. `requirements.txt` pins pathway core plus the
> few optional libs `xpacks.llm` actually calls. See the note in that file.

## Configuration

All knobs live in [`realtime_rag/config.py`](realtime_rag/config.py) with safe
defaults; override any of them in `.env` (see [`.env.example`](.env.example)).
Only `GROQ_API_KEY` is required for real answers; `NEWSAPI_KEY` is optional
(stock quotes still stream without it). The default profile runs embeddings
fully offline — the most reliable setup for a live demo.

## Proving the dynamism (the 35% criterion)

With the backend running:

```bash
scripts/demo_dynamism.sh
```

It performs, end to end, with measured latencies:

1. **Baseline** — ask *"What is Project Zephyr?"* → not found.
2. **ADD** `data/docs/zephyr.txt` → after ~2–5s the answer describes Project Zephyr.
3. **MODIFY** the same file (→ "cancelled") → the answer now says cancelled.
4. **DELETE** the file (`rm`) → the answer no longer reflects it; `file_count` drops.
5. **Channel B** — ask for the latest AAPL price twice ~30s apart → the value
   changes and only one current quote exists (upsert, not accumulation).

No process is restarted at any point. The same flow can be driven from the
Streamlit sidebar (upload / delete document) for the video demo.

Target latency: **data change → reflected in the answer in 2–5s**
(vs. 2–50s in the previous design); warm query round-trip **< 1.5s**.

## Tests

```bash
pip install pytest pytest-timeout
pytest
```

Covers: filesystem add/modify/delete changes retrieved chunks with no restart;
the finance connector emits correct `+1` / retract-then-insert (upsert) / `-1`
diffs; end-to-end answer changes after modify/delete (no stale cache); async LLM
timeout degrades gracefully; config/back-end switches.

## Project layout

```


ui/                   Streamlit dashboard on the pathway RAGClient
data/docs/            Channel A watched folder (seed_* fixtures included)
scripts/              demo_dynamism.sh, smoke_query.sh
tests/                pytest suite
Dockerfile, docker-compose.yml
```

The legacy `app.py` / `dashboard.py` from the previous design were replaced;
they remain in git history for reference.
