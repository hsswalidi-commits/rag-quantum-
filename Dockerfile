FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8080

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY knowledge_base ./knowledge_base
COPY main.py ./

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /app /home/app

ENV PATH="/app/.venv/bin:$PATH"

USER app
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]