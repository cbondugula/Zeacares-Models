FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: spaCy model download removed — NERExtractor(use_model=False) never uses it
# and it added ~800 MB to the image. Re-enable only if you flip use_model=True.

# Copy application code
COPY src/ ./src/
# Create results dir; if a real results/ folder is needed at build time, copy it
# in a separate explicit COPY (Docker COPY does NOT understand shell syntax).
RUN mkdir -p /app/results /app/model_cache

# Non-root user for security
RUN useradd -m zeacares && chown -R zeacares:zeacares /app
USER zeacares

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CRITICAL FIX vs original (--workers 2 caused repeated PubMedBERT loads + OOM → 504):
#   * --workers 1: only one process loads the ~1.5 GB transformer + FAISS index.
#   * Scale concurrency with --threads / async, NOT extra workers, until you have
#     enough RAM (~3 GB per worker) to safely fork.
#   * --timeout 120: model cold-start can take >60 s on CPU; default 30s killed it.
#   * lifespan handler in src/api/main.py already loads model ONCE per process.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
