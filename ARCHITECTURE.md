# Architecture

## 1. Overview

FinRAG is a Retrieval-Augmented Generation system built on **Pathway**'s native
incremental `DocumentStore`. It serves answers over two live data channels:

- **Channel A** — a watched documents folder (`data/docs/*.txt`, `*.pdf`).
- **Channel B** — a live finance/news feed (yfinance quotes + NewsAPI articles).

The headline guarantee: **every add, modify, or delete on either channel
propagates — through parsing, chunking, embedding, and the vector index — to
the very next answer, with no restart and no batch re-indexing**. End-to-end
latency from data event to changed answer is a few seconds (embed + index of
one changed file); warm query round-trip is under ~1.5 s.

## 2. System diagram

```
                          ┌──────────────────────────────────────────────────┐
  Channel A               │                  PATHWAY ENGINE                  │
  data/docs/*.txt,*.pdf   │   (differential dataflow — every edge carries    │
  ───(fs connector,       │    (row, +1/-1, time) diffs; recompute is        │
      streaming,          │    incremental and minimal)                      │
      +1/-1/modify)──────▶│                                                  │
                          │   pw.io.fs.read ─┐                               │
  Channel B               │                  ├─▶ DocumentStore               │
  yfinance + NewsAPI      │   finance feed ──┘     parser → splitter →       │
  ───(custom Python       │   (pw.io.python,        embedder → KNN index     │
      UPSERT connector,   │    UPSERT by PK)        (incremental)            │
      keyed by symbol /   │                                │                 │
      sha1(url))─────────▶│                                ▼                 │
                          │              BaseRAGQuestionAnswerer             │
   HTTP question ─────────┼─▶ rest_connector ─▶ retrieve retrieve_topk ─▶    │
   (POST /v2/answer)      │      rerank → search_topk ─▶ async LiteLLM (Groq)│
   ◀──────────────────────┼──── {response, context_docs} ◀───────────────────┤
                          └──────────────────────────────────────────────────┘
        ▲
        │ HTTP (pathway RAGClient — no SQLite, no CSV, no polling)
   Streamlit UI (ui/ui.py): chat • market snapshot • upload/delete docs
```

## 3. Components

### 3.1 Ingestion — Channel A (documents)

`pw.io.fs.read` tails `data/docs/` in streaming mode. The filesystem connector
emits a stream of diffs: `(row, +1, t)` when a file appears, `(-1, +1)` when it
is rewritten, `(-1)` when it is removed. The UI writes uploads as a temp file
then atomically `os.replace()`s into `data/docs/`, so the watcher never sees a
half-written file.

### 3.2 Ingestion — Channel B (live market & news)

`realtime_rag/connectors/finance_feed.py` runs a custom `pw.io.python.ConnectorSubject`
that polls `market_clients` on a fixed interval. Quotes and news rows are
**upserts** keyed by:

- `symbol` for quotes (one row per ticker, replaced on each poll).
- `sha1(url)` for news articles (one row per article URL).

Re-emitting the same key triggers an engine-level retract + insert, so a new
AAPL price is a true **modification** — the stale quote leaves the index
instead of accumulating forever.

A content-hash dedup (`NEWS_DEDUP=true`, default) compares the *material*
fields of each row (excluding the polling timestamp) and skips the emit when
nothing has changed, so unchanged quotes/articles do not waste embeddings.

Every network call is wrapped in a hard timeout + full exception containment —
the connector thread never raises; upstream outages just stop adding new rows.

### 3.3 DocumentStore — parse → split → embed → index

Both channels feed Pathway's `xpacks.llm.DocumentStore`. The pipeline:

1. **Parser** (`PARSER_BACKEND=utf8` default; `docling` optional for PDFs).
   A defensive wrapper drops un-parseable files instead of poisoning the graph.
2. **Splitter** (`SPLITTER_BACKEND=recursive` default).
   `RecursiveSplitter` (token units via `cl100k_base`) keeps a `CHUNK_OVERLAP`
   (default 80-token) window between adjacent chunks, so a fact straddling a
   `CHUNK_MAX_TOKENS` (default 400) boundary is not split across two chunks
   that each retrieve poorly. `SPLITTER_BACKEND=token` restores hard-cut
   `TokenCountSplitter` with no overlap.
3. **Embedder** (`EMBEDDER_BACKEND=local` default).
   `BAAI/bge-small-en-v1.5` runs fully offline. `openai` and `gemini` are
   switchable via config; the local default is the most reliable demo profile.
4. **Index** (`INDEX_BACKEND=bruteforce` default).
   Brute-force KNN is exact and gives trivially-correct `+1/-1` incremental
   delete behaviour. `usearch` (HNSW) is the documented swap for corpora above
   ~50k chunks.

### 3.4 Question answering — retrieve → rerank → LLM

`BaseRAGQuestionAnswerer` runs a two-stage retrieval:

- **Vector recall** — the indexer returns a wide pool of `RETRIEVE_TOPK`
  (default 20) candidates by similarity.
- **Cross-encoder rerank** (`RERANKER_BACKEND=flashrank` default) — a
  cross-encoder reorders the pool and the top `SEARCH_TOPK` (default 6) go to
  the LLM. `flashrank` is the default because its TinyBERT model scores in
  ~milliseconds — a quality lift with negligible latency. `cross_encoder`
  swaps in the stronger, heavier ms-marco-MiniLM; `none` is pure vector order
  (single-stage). A missing optional dep or signature drift degrades to `none`
  (logged), never crashes.

Reranking is wired for `RAG_MODE=base` only — `AdaptiveRAGQuestionAnswerer`
has no reranker hook upstream, so adaptive logs `reranker_ignored_adaptive`
and proceeds single-stage.

The LLM (Groq `llama-3.3-70b-versatile` via LiteLLM by default) runs as a
Pathway async UDF (`LiteLLMChat(async_mode=...)`) with bounded `capacity` and
exponential-backoff retries. A slow LLM call cannot stall the dataflow worker.

### 3.5 Serving — REST + Streamlit UI

The Pathway REST connector exposes:

- `POST /v2/answer` — the RAG endpoint; accepts an optional `filters` jmespath
  string for metadata-scoped retrieval (see [§7 Configuration](#7-configuration-surface)).
- `POST /v1/statistics` — health + indexer stats; doubles as the Docker
  healthcheck.

The Streamlit UI (`ui/ui.py`) talks to the REST API through Pathway's
`RAGClient` — no SQLite, no CSV, no polling. It also supports document
upload/delete and a search-scope filter mapped to a jmespath expression.

## 4. How incremental updates propagate

Pathway is a **differential dataflow** engine. A connector does not emit "the
current state"; it emits a stream of diffs:

- `(row, +1, t)` for an insertion.
- `(row, -1, t)` for a retraction.

Every operator (parser, splitter, embedder, KNN index) is a *standing
computation* defined once over these streams.

| Event | Diffs emitted | What recomputes |
|---|---|---|
| `touch new.txt` | `+1` for the new row | parse → split → embed → index inserts the new chunk vectors |
| Edit existing file | `-1` for the old row, `+1` for the new row | old chunks retracted from the index; new chunks inserted |
| `rm file.txt` | `-1` for the row | retraction propagates; the index removes exactly that file's chunk vectors |
| New AAPL quote | `-1` (old `AAPL` row) + `+1` (new row) | stale quote leaves the index; new quote replaces it |

There is **no polling loop and no "reindex" job**. Liveness is a property of
the computation graph, not a scheduled task.

Concurrent-delete safety: differential dataflow is consistent per processing
timestamp — an in-flight query observes either the pre-delete or post-delete
snapshot, never a torn state.

## 5. Caching & freshness

Caching is delegated to Pathway's content-keyed `cache_strategy` at the
embedder/LLM UDF level. The cache key is derived from the *actual input
content*, so when retrieved context changes (because a document changed) the
LLM prompt changes, the cache key changes, and a fresh answer is computed.
A stale answer is unreachable by construction.

Persistence: Pathway's filesystem persistence/UDF cache lives in `./Cache`,
which lets the process restart without recomputing embeddings for unchanged
content. The connector frontier is also persisted.

> **Operational note — clear `./Cache` when you change chunking.** On restart
> Pathway recovers the connector frontier and *seeks the fs source past
> already-ingested files*, so changing `SPLITTER_BACKEND` / `CHUNK_*` while an
> old `./Cache` exists leaves Channel A documents chunked by the *previous*
> splitter (Channel B keeps flowing because it is a live connector). Deleting
> `./Cache` (it is gitignored and regenerated) forces a clean re-ingest under
> the new chunking. This is expected persistence behaviour, not a bug.

## 6. Resilience

- **Async, non-blocking LLM** — bounded `capacity` and exponential-backoff
  retries; a slow Groq call never stalls a dataflow worker.
- **Upstream outages** — `market_clients` wrap every network call in a hard
  timeout + total exception containment and never raise; the connector thread
  keeps looping; the index keeps serving existing data.
- **Bad/partial files** — atomic `os.replace()` for uploads + a defensive
  parse wrapper that drops un-parseable files; a max-size guard rejects
  oversize uploads.
- **Empty/scanned PDFs** — the UI's `add_document()` rejects PDFs with no
  extractable text with a user-visible warning, instead of silently embedding
  a blank document.
- **Consistency under concurrent delete** — see [§4](#4-how-incremental-updates-propagate).
- **Persistence** — `./Cache` lets the process restart without recomputing
  embeddings for unchanged content.
- **Observability** — one-JSON-object-per-line structured logs with stable
  `event=` keys and `latency_ms`; `POST /v1/statistics` doubles as the
  Docker healthcheck.

## 7. Configuration surface

Everything is switchable via `.env` / `realtime_rag/config.py` with safe
defaults. The default profile (local `BAAI/bge-small-en-v1.5` embeddings +
brute-force KNN + Groq LLM + FlashRank rerank) needs **no API key except
`GROQ_API_KEY`** and runs fully offline for embeddings.

### Core backends

| Env var | Default | Options |
|---|---|---|
| `EMBEDDER_BACKEND` | `local` | `local` \| `openai` \| `gemini` |
| `LLM_BACKEND` | `groq` | `groq` \| `openai` |
| `RAG_MODE` | `base` | `base` \| `adaptive` |
| `INDEX_BACKEND` | `bruteforce` | `bruteforce` \| `usearch` |
| `PARSER_BACKEND` | `utf8` | `utf8` \| `docling` |

### Retrieval quality & UX (all one-env-var reversible)

| Env var | Default | Effect |
|---|---|---|
| `SPLITTER_BACKEND` | `recursive` | `recursive` = sentence-aware overlap chunking; `token` = hard cuts |
| `CHUNK_OVERLAP` | `80` | token overlap between adjacent chunks (recursive only) |
| `CHUNK_MAX_TOKENS` | `400` | max tokens per chunk |
| `RERANKER_BACKEND` | `flashrank` | `flashrank` (tiny, fast) \| `cross_encoder` (stronger) \| `none` (vector-only) |
| `RETRIEVE_TOPK` | `20` | wide vector-recall pool before rerank |
| `SEARCH_TOPK` | `6` | final chunks sent to the LLM (post-rerank) |
| `NEWS_DEDUP` | `true` | skip re-embedding a quote/news row whose material content is unchanged |
| `CHAT_HISTORY_TURNS` | `3` | prior turns the UI prepends for follow-ups (0 = off) |

### UI

The Streamlit sidebar exposes a **Search Scope** control (All / Documents only /
Live market only) that maps to a jmespath metadata filter on the `source` key
(only Channel B rows carry `source=finance_feed`). Multi-turn follow-ups are
implemented by prepending the last `CHAT_HISTORY_TURNS` turns client-side; the
backend stays stateless.

## 8. Scaling & extension

- **Larger corpora** — switch `INDEX_BACKEND=usearch` for HNSW sub-linear
  retrieval when the corpus exceeds ~50k chunks. Brute-force is the default
  because its exact `+1/-1` incremental-delete correctness is cleaner to
  verify.
- **Stronger / cheaper rerank** — swap `RERANKER_BACKEND=cross_encoder` for a
  heavier reranker, or `none` to drop the second stage entirely, with no code
  change.
- **More data sources** — `DocumentStore` accepts multiple sources; add a
  Google Drive / SharePoint / S3 / Kafka connector with no change to the RAG
  or serving layers.
- **Horizontal scale** — Pathway supports multi-worker / distributed execution;
  the REST layer is stateless.
- **Domain change** — nothing in the pipeline is finance-specific except
  Channel B's fetchers; point `data/docs/` at any corpus to repurpose it.

## 9. Production hardening (intentionally out of demo scope)

The demo deliberately ships the **clean RAGClient ↔ Pathway REST** shape with
no auth/rate-limit proxy in front of it, so the incremental-dataflow story
stays the focus and there is no extra process/port to explain. For a real
deployment the following sits at the edge (reverse proxy / API gateway),
*not* in the Pathway process — keeping the engine single-responsibility:

- **Rate limiting** — per-client token bucket (e.g. 60 req/min burst 10),
  keyed by API key then client IP, returning `429` + `Retry-After`. Belongs in
  the gateway (nginx `limit_req`, Envoy, or a slowapi/ASGI middleware on a
  thin FastAPI reverse proxy) so the dataflow worker is never the throttling
  point.
- **Authentication** — an `Authorization: Bearer <key>` / `X-API-Key` header
  checked at the gateway; unauthenticated requests never reach the engine.
  Keys issued per consumer for revocation and per-key quotas.
- **TLS + network** — terminate TLS at the gateway; the Pathway port stays
  bound to localhost / the internal network only.
