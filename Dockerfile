# syntax=docker/dockerfile:1.7
FROM python:3.14-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.17

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
# Bake the public embedding model into the image. No model download on startup.
RUN uv run --no-sync hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

FROM python:3.14-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /home/app app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /opt/huggingface /opt/huggingface
COPY --chown=app:app main.py /app/main.py

USER 10001:10001
EXPOSE 8080

# Cloud Run injects PORT. One worker avoids duplicating the large model per process.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT}\" --workers 1 --proxy-headers"]
