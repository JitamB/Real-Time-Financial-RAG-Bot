#!/usr/bin/env bash
# Smoke test: health + a single question against the running backend.
# Usage: scripts/smoke_query.sh "What is in the seed company brief?"
set -euo pipefail

HOST="${RAG_HOST:-localhost}"
PORT="${RAG_PORT:-8000}"
BASE="http://${HOST}:${PORT}"
Q="${1:-What is Acme Robotics revenue?}"

echo "==> POST ${BASE}/v1/statistics (health)"
curl -fsS -X POST "${BASE}/v1/statistics" -H 'Content-Type: application/json' -d '{}' \
  | sed 's/^/    /' || {
  echo "    backend not reachable on ${BASE}" >&2
  exit 1
}
echo

echo "==> POST ${BASE}/v2/answer"
curl -fsS -X POST "${BASE}/v2/answer" \
  -H 'Content-Type: application/json' \
  -d "$(printf '{"prompt": %s, "return_context_docs": true}' "$(printf '%s' "$Q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
  | python3 -m json.tool | sed 's/^/    /'
