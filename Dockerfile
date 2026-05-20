# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Real-Time Incremental RAG on Pathway.
# Base image ships the Pathway engine; we add the xpacks.llm + ML stack.
# ---------------------------------------------------------------------------
FROM pathwaycom/pathway:latest

WORKDIR /app

# DoclingParser needs OpenCV + Tesseract for PDF/table/OCR parsing.
# (Switch PARSER_BACKEND=unstructured to build a leaner image without these.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-opencv tesseract-ocr tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default local embedding model into the image layer so the
# first query is fast and the container works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY realtime_rag/ ./realtime_rag/
COPY ui/ ./ui/
COPY data/docs/ ./data/docs/
COPY styles.css ./styles.css

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Pathway REST endpoints are POST (rest_connector). Generous start-period:
# cold embedder load + graph build before the server accepts requests.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=6 \
    CMD curl -fsS -X POST "http://localhost:${PORT}/v1/statistics" \
        -H 'Content-Type: application/json' -d '{}' || exit 1

CMD ["python", "-m", "realtime_rag.app"]
