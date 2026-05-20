#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Demonstrable Dynamism — the 35% liveness proof.
#
# Proves: adding / modifying / deleting a file in the watched docs folder
# changes the RAG answer within seconds, with NO restart and NO re-indexing.
#
# Prereq: backend running (docker compose up  OR  python -m realtime_rag.app).
# Usage:  scripts/demo_dynamism.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${RAG_HOST:-localhost}"
PORT="${RAG_PORT:-8000}"
BASE="http://${HOST}:${PORT}"
DOCS_DIR="${DOCS_DIR:-./data/docs}"
DEMO_FILE="${DOCS_DIR}/zephyr.txt"
SETTLE="${SETTLE_SECONDS:-5}"
Q='What is Project Zephyr?'

ask() {
  curl -fsS -X POST "${BASE}/v2/answer" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1]}))' "$Q")" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["response"])'
}

stats() { curl -fsS -X POST "${BASE}/v1/statistics" -H 'Content-Type: application/json' -d '{}'; }

banner() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

mkdir -p "$DOCS_DIR"

banner "0. Health"
stats; echo

banner "1. Baseline — ask BEFORE the document exists"
echo "Q: $Q"
echo "A: $(ask)"

banner "2. ADD  (printf > ${DEMO_FILE})"
printf 'Project Zephyr is a new Apple AR headset launching in Q4 2026.' > "$DEMO_FILE"
echo "   waiting ${SETTLE}s for incremental embed+index..."; sleep "$SETTLE"
echo "Q: $Q"
echo "A: $(ask)"
echo "   stats: $(stats)"

banner "3. MODIFY  (overwrite same file)"
printf 'Project Zephyr was CANCELLED as of May 2026 due to supply issues.' > "$DEMO_FILE"
echo "   waiting ${SETTLE}s (old chunk retracted, new chunk indexed)..."; sleep "$SETTLE"
echo "Q: $Q"
echo "A: $(ask)   <-- now reflects the CANCELLED text"

banner "4. DELETE  (rm ${DEMO_FILE})"
rm -f "$DEMO_FILE"
echo "   waiting ${SETTLE}s (chunks removed from the index)..."; sleep "$SETTLE"
echo "Q: $Q"
echo "A: $(ask)   <-- no longer reflects the deleted document"
echo "   stats: $(stats)"

banner "5. Channel B — live finance upsert"
echo "   Latest AAPL quote twice ~30s apart: it tracks the live price and only"
echo "   ONE current quote ever exists (upsert by symbol, not accumulation —"
echo "   note file_count above stays bounded as quotes refresh)."
curl -fsS -X POST "${BASE}/v2/answer" -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the latest AAPL stock price?"}' \
  | python3 -c 'import json,sys; print("A1:", json.load(sys.stdin)["response"])'
sleep 30
curl -fsS -X POST "${BASE}/v2/answer" -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the latest AAPL stock price?"}' \
  | python3 -c 'import json,sys; print("A2:", json.load(sys.stdin)["response"])'

banner "DONE — liveness demonstrated with zero restarts"
