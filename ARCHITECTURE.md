# Architecture

## 1. The problem with the old design

The previous implementation (`app.py`, kept in git history for reference) was **not a RAG system**. It collapsed every document ever seen into a single unbounded Python tuple under a constant Pathway key and then did O(n) substring matching over that whole tuple on every query. Consequences:

- No embeddings, no vector index, no chunking → slow and semantically blind.
- Modifications and deletions were structurally impossible — the tuple only grew, and a file-hash side-effect written *inside a UDF* returned a stale timestamp for re-added content.
- A process-global LRU cache keyed by `(timestamp, query)` served stale answers even after the underlying data changed.
- The UI bridge was `CSV → pandas re-read → SQLite → poll every 1s up to 45s` (2–50s latency) and a maintenance thread rewrote the very CSV the Pathway reader was tailing.

That design **cannot** satisfy the hackathon's central criterion ("Demonstrable Dynamism", 35%).

## 2. The new design in one sentence

We replace the hand-rolled retrieval with **Pathway's native incremental `DocumentStore`** so that every add / modify / delete on the data sources propagates automatically — through parsing, chunking, embedding, and the vector index — to the next answer, with **no restart and no batch re-indexing**.

## 3. Data flow

```
                          ┌─────────────────────────────────────────────────┐
  Channel A               │                 PATHWAY ENGINE                   │
  data/docs/*.txt,*.pdf   │  (differential dataflow — every edge carries     │
  ───(fs connector,       │   (row, +1/-1, time) diffs, recompute is         │
      streaming,          │    incremental and minimal)                      │
      +1/-1/modify)──────▶│                                                  │
                          │   pw.io.fs.read ─┐                               │
  Channel B               │                  ├─▶ DocumentStore               │
  yfinance + NewsAPI      │   finance feed ──┘     parser → splitter →        │
  ───(custom Python       │   (pw.io.python,        embedder → KNN index      │
      UPSERT connector,    │    UPSERT by PK)        (incremental)            │
      keyed by symbol /   │                                │                 │
      sha1(url))──────────▶│                                ▼                 │
                          │              BaseRAGQuestionAnswerer             │
   HTTP question ─────────┼────▶ rest_connector ─▶ retrieve top-k ─▶ async   │
   (POST /v2/answer)      │                          LiteLLM (Groq)          │
   ◀──────────────────────┼──── {response, context_docs} ◀───────────────────┤
                          └─────────────────────────────────────────────────┘
        ▲
        │ HTTP (pathway RAGClient — no SQLite, no CSV, no polling)
   Streamlit UI (ui/ui.py): chat • market snapshot • upload/delete docs
```

## 4. Why deletion/modification "just works" (the core idea)

Pathway is a **differential dataflow** engine. A connector does not emit "the current state"; it emits a stream of **diffs**: `(row, +1, t)` for an insertion and `(row, -1, t)` for a retraction. Every operator (parser, splitter, embedder, KNN index) is a *standing computation* defined once over these streams.

- **Add** a file → the filesystem connector emits `+1` for that file's row → it flows through parse → split → embed → the index inserts exactly those new chunk vectors.
- **Modify** a file → the connector emits `-1` for the old row and `+1` for the new row → the old chunks are retracted from the index and the new chunks inserted. Only the changed file is recomputed.
- **Delete** a file (`rm`) → the connector emits `-1` → the retraction propagates through the same operators → the index removes exactly that file's chunk vectors. Nothing else recomputes.

There is **no polling loop and no "reindex" job**. Liveness is a property of the computation graph, not a scheduled task. This is what makes the latency from a data event to a changed answer a few seconds (embed + index of one changed file), not minutes.

The same mechanism powers **Channel B**: the finance/news connector is an *upsert* source keyed by `symbol` (quotes) and `sha1(url)` (news). Re-emitting the same key replaces the previous row (engine-level retract + insert), so a new AAPL price is a true *modification* — the stale quote leaves the index instead of accumulating forever.

## 5. No stale answers

The broken `(timestamp, query)` LRU cache is **removed entirely**. Caching is delegated to Pathway's content-keyed `cache_strategy` at the embedder/LLM UDF level: the cache key is derived from the *actual input content*. If the retrieved context changes (because a document changed), the LLM prompt changes, so the cache key changes, so a fresh answer is computed. A stale answer is therefore unreachable by construction.

## 6. Resilience & production concerns

- **Async, non-blocking LLM**: the LLM runs as a Pathway async UDF (`LiteLLMChat(async_mode=...)`) with bounded `capacity` and exponential-backoff retries. A slow Groq call never stalls the dataflow worker (the root flaw of the old synchronous in-UDF call).
- **Upstream outages**: `market_clients` wrap every network call in a hard timeout + total exception containment and never raise — the connector thread keeps looping; the index keeps serving existing data.
- **Bad/partial files**: the UI writes a temp file then atomically `os.replace()`s it into `data/docs/`, so the watcher never sees a half-written file; a defensive parse wrapper drops un-parseable files instead of poisoning the graph; a max-size guard rejects oversize uploads.
- **Consistency under concurrent delete**: differential dataflow is consistent per processing timestamp — an in-flight query observes either the pre-delete or post-delete snapshot, never a torn state.
- **Persistence**: Pathway's filesystem persistence/UDF cache (`./Cache`) lets the process restart without recomputing embeddings for unchanged content.
- **Observability**: one-JSON-object-per-line structured logs with stable `event=` keys and `latency_ms`; `GET /v1/statistics` doubles as a health/liveness probe (used by the Docker healthcheck).

## 7. Configuration surface

Everything is switchable via `.env` / `realtime_rag/config.py` with safe defaults:
`embedder_backend` (local | openai | gemini), `llm_backend` (groq | openai), `rag_mode` (base | adaptive), `index_backend` (bruteforce | usearch), `parser_backend` (docling | unstructured), `search_topk`, `chunk_max_tokens`, poll intervals, host/port, timeouts. The default profile (local `BAAI/bge-small-en-v1.5` embeddings + brute-force KNN + Groq LLM) needs **no API key except Groq** and runs fully offline for embeddings — the most reliable profile for a live demo.

## 8. Scaling & extension

- Swap `index_backend=usearch` for an HNSW index (sub-linear retrieval) when the corpus grows.
- The `DocumentStore` accepts multiple sources; add a Google Drive / SharePoint / S3 / Kafka connector with no change to the RAG or serving layers.
- Horizontal scale: Pathway supports multi-worker/distributed execution; the REST layer is stateless.
- Domain change: nothing in the pipeline is finance-specific except Channel B's fetchers — point `data/docs/` at any corpus to repurpose it.
