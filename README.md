# FinRAG - Real-Time Incremental Financial RAG

A Retrieval-Augmented Generation system that **thinks in real time**. Add,
modify, or delete a document in the watched folder — or let the live
market/news feed update — and the chatbot's answer to a relevant question
changes within **seconds**, with **no manual restart and no batch
re-indexing**.

Built on [Pathway](https://pathway.com)'s differential-dataflow engine and
its native incremental `DocumentStore`.

## Table of contents

- [Highlights](#highlights)
- [Architecture at a glance](#architecture-at-a-glance)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Demo: proving the dynamism](#demo-proving-the-dynamism)
- [Tests & evaluation](#tests--evaluation)
- [Project layout](#project-layout)
- [Tech stack](#tech-stack)

## Highlights

| Capability | Typical RAG | FinRAG |
|---|---|---|
| New document | re-run an indexing batch job | indexed incrementally in seconds, automatically |
| Edited document | stale until next batch | old chunks retracted, new chunks indexed, automatically |
| **Deleted document** | usually **never** removed | chunks removed from the vector index, automatically |
| Live market data | appended forever (grows unbounded) | **upserted** by key — only the latest state is retrieved |
| Stale answers | LRU cache can serve outdated replies | impossible — caching is content-keyed |
| Target latency | | data change → reflected in answer in **2–5 s**; warm round-trip **< 1.5 s** |

See [ARCHITECTURE.md](ARCHITECTURE.md) for *why* deletion "just works"
(differential dataflow), the data-flow diagram, the resilience model, and the
scaling notes.

## Architecture at a glance

```
Channel A: data/docs/  ──(fs streaming, +1/-1/modify)───┐
                                                        ├─▶ Pathway DocumentStore
Channel B: yfinance+NewsAPI ──(Python UPSERT connector)─┘   parse→split→embed→KNN
                                                                     │
HTTP  POST /v2/answer ─▶ rest_connector ─▶ retrieve(20) ─▶ rerank → top-6 ─▶ async LiteLLM (Groq)
Streamlit UI ──(pathway RAGClient over HTTP; no SQLite/CSV/polling)──┘
```

- **Engine** — Pathway 0.30.x (`xpacks.llm` `DocumentStore`, incremental KNN).
- **Embeddings** — local `BAAI/bge-small-en-v1.5` by default (offline, no
  quota); switchable to OpenAI / Gemini via config.
- **Retrieval** — sentence-aware overlap chunking + a two-stage
  vector-recall → FlashRank cross-encoder rerank (both one-env-var reversible).
- **LLM** — Groq `llama-3.3-70b-versatile` via LiteLLM, called as a
  non-blocking async UDF with retries.
- **Serving** — Pathway REST API; Streamlit dashboard talks to it directly via
  `RAGClient` (no SQLite, no CSV, no polling).

## Quick start

### Option A — Docker (recommended)

```bash
cp .env.example .env          # then add your GROQ_API_KEY (and optional NEWSAPI_KEY)
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
```

#### &emsp; Terminal 1 — backend (REST :8000)
```bash
python -m realtime_rag.app
```

#### &emsp; Terminal 2 — Streamlit UI (:8501)
```bash
streamlit run ui/ui.py
```

> The `pathway[xpack-llm]` *extra* is intentionally **not** used — it makes
> pip fail with `resolution-too-deep`. `requirements.txt` pins pathway core
> plus the few optional libs `xpacks.llm` actually calls. See the note in that
> file.

## Configuration

All knobs live in [`realtime_rag/config.py`](realtime_rag/config.py) with safe
defaults; override any of them in `.env` (see [`.env.example`](.env.example)).
Only `GROQ_API_KEY` is required for real answers; `NEWSAPI_KEY` is optional
(stock quotes still stream without it). The default profile runs embeddings
fully offline.

### Core backends

| Env var | Default | Options |
|---|---|---|
| `EMBEDDER_BACKEND` | `local` | `local` \| `openai` \| `gemini` |
| `LLM_BACKEND` | `groq` | `groq` \| `openai` |
| `RAG_MODE` | `base` | `base` \| `adaptive` |
| `INDEX_BACKEND` | `bruteforce` | `bruteforce` \| `usearch` |
| `PARSER_BACKEND` | `utf8` | `utf8` \| `docling` |

### Retrieval quality & UX

| Env var | Default | Effect |
|---|---|---|
| `SPLITTER_BACKEND` | `recursive` | sentence-aware overlap chunking (`token` = hard cuts) |
| `CHUNK_OVERLAP` | `80` | token overlap between adjacent chunks |
| `RERANKER_BACKEND` | `flashrank` | two-stage rerank: `flashrank` (fast) \| `cross_encoder` \| `none` |
| `RETRIEVE_TOPK` | `20` | wide vector pool before the reranker narrows to `SEARCH_TOPK` |
| `NEWS_DEDUP` | `true` | skip re-embedding unchanged quote/news rows |
| `CHAT_HISTORY_TURNS` | `3` | prior turns the UI prepends for follow-ups (0 = off) |

The Streamlit sidebar adds a **Search Scope** selector (All / Documents only /
Live market only) and the chat supports **multi-turn follow-ups** (the last
`CHAT_HISTORY_TURNS` turns are prepended client-side; the backend stays
stateless). See [ARCHITECTURE.md §3.4 / §7](ARCHITECTURE.md) for the
two-stage retrieval rationale.

## Demo: proving the dynamism

With the backend running:

```bash
scripts/demo_dynamism.sh
```

It performs, end to end, with measured latencies:

1. **Baseline** — ask *"What is Project Zephyr?"* → not found.
2. **ADD** `data/docs/zephyr.txt` → after ~2–5 s the answer describes Project Zephyr.
3. **MODIFY** the same file (→ "cancelled") → the answer now says cancelled.
4. **DELETE** the file (`rm`) → the answer no longer reflects it; `file_count` drops.
5. **Channel B** — ask for the latest AAPL price twice ~30 s apart → the value
   changes and only one current quote exists (upsert, not accumulation).

No process is restarted at any point. The same flow can be driven from the
Streamlit sidebar (upload / delete document) for the video demo.

## Tests & evaluation

### Unit tests

```bash
pip install pytest pytest-timeout
pytest
```

Covers: filesystem add/modify/delete changes retrieved chunks with no restart;
the finance connector emits correct `+1` / retract-then-insert (upsert) / `-1`
diffs *and* content-hash dedup (unchanged rows are not re-embedded); end-to-end
answer changes after modify/delete (no stale cache); async LLM timeout degrades
gracefully; config / back-end switches (splitter, reranker, dedup, history).

### Answer-quality eval

Needs a live backend; **not** in the default `pytest` run.

```bash
python -m realtime_rag.app                  # terminal 1
python scripts/eval.py                      # keyword recall over a grounded golden set
python scripts/eval.py --judge              # + LLM faithfulness (reuses the Groq model)
```

A dependency-free harness — no RAGAS — over
[`tests/golden/seed_glossary_qa.json`](tests/golden/seed_glossary_qa.json),
grounded only in a committed `seed_*` fixture so it is reproducible.

## Project layout

```
Real-Time-Financial-RAG-Bot/
├── realtime_rag/                  # Pathway pipeline (the engine)
│   ├── app.py                     # entrypoint: builds the graph, serves REST
│   ├── config.py                  # pydantic settings, all env-var knobs
│   ├── connectors/
│   │   ├── document_source.py     # Channel A: pw.io.fs.read on data/docs/
│   │   ├── finance_feed.py        # Channel B: yfinance + NewsAPI UPSERT subject
│   │   └── market_clients.py      # network calls (timeouts + exception containment)
│   ├── pipeline/
│   │   ├── document_store.py      # parser + splitter + embedder + KNN index
│   │   └── rag.py                 # reranker + BaseRAGQuestionAnswerer + LLM UDF
│   ├── server/
│   │   └── rest_app.py
│   └── observability/
│       └── logging.py             # structured JSON logs, stable event= keys
│
├── ui/                            # Streamlit dashboard
│   ├── ui.py                      # chat + market snapshot + upload/delete
│   └── components.py              # ask() helper, doc upload, scope filter
│
├── data/
│   └── docs/                      # Channel A watched folder
│       └── seed_market_glossary.txt
│
├── scripts/
│   ├── demo_dynamism.sh           # the headline add/modify/delete liveness demo
│   ├── smoke_query.sh             # one-shot curl against /v2/answer
│   └── eval.py                    # slim answer-quality harness (no RAGAS)
│
├── tests/                         # pytest suite (graph assembly + connectors)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_document_store.py
│   ├── test_finance_feed_upsert.py
│   ├── test_market_clients.py
│   └── golden/
│       └── seed_glossary_qa.json  # Q/A pairs for scripts/eval.py
│
├── ARCHITECTURE.md
├── README.md                      # this file
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                 # package metadata + deps
├── requirements.txt
├── .env.example                   # documented env-var template
├── .dockerignore
├── .gitignore
└── styles.css
```

Created at runtime (gitignored):

- `Cache/` — Pathway's connector frontier + embedding/UDF cache.
- `flashrank_cache/` — FlashRank reranker model cache.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Dataflow engine | Pathway 0.30.x | Native incremental `DocumentStore`; differential `+1/-1` diffs make add/modify/delete trivially correct |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Offline, no quota, fast; OpenAI / Gemini swap available |
| Vector index | brute-force KNN | Exact; trivially correct incremental delete. `usearch` HNSW available for >~50k chunks |
| Splitter | `RecursiveSplitter` (token, with overlap) | Avoids splitting facts across chunks; `TokenCountSplitter` available as fallback |
| Reranker | FlashRank TinyBERT (default) | ~ms latency; quality lift over pure vector order. `cross_encoder` (ms-marco-MiniLM) and `none` available |
| LLM | Groq `llama-3.3-70b-versatile` via LiteLLM | Fast inference; async UDF with bounded capacity + retries |
| API | Pathway REST connector | `POST /v2/answer`, `POST /v1/statistics`; stateless |
| UI | Streamlit + `pathway.RAGClient` | Direct HTTP to the REST API; no SQLite / CSV / polling bridge |
| Packaging | Docker + docker-compose | Backend + UI in one `up`; model weights baked into the image |
