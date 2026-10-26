from __future__ import annotations

import logging
import os
import secrets
import time
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag.core import (
    MAX_HISTORY_TURNS,
    ConversationMemory,
    ConversationTurn,
    answer_question,
    load_clients,
)

logger = logging.getLogger(__name__)


class TurnPayload(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    rewritten_question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=3000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    history: list[TurnPayload] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    history: list[TurnPayload]


MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "12000"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}


def _authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    token = os.getenv("API_ACCESS_TOKEN")
    # Local development may run without a token. Cloud Run always requires one.
    if not token and not os.getenv("K_SERVICE"):
        return
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API access is not configured.",
        )
    scheme, _, supplied_token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied_token, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        llm_client, embedding_model, qdrant_client = await run_in_threadpool(load_clients)
    except Exception:
        logger.exception("API initialization failed")
        raise

    application.state.llm_client = llm_client
    application.state.embedding_model = embedding_model
    application.state.qdrant_client = qdrant_client
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        close = getattr(qdrant_client, "close", None)
        if close:
            await run_in_threadpool(close)


docs_enabled = os.getenv("ENABLE_DOCS", "0").lower() in {"1", "true", "yes"}
app = FastAPI(
    title="Quantum Knowledge Assistant API",
    version="1.0.0",
    description="Arabic and English retrieval-augmented question answering API.",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_enabled else None,
)


@app.get("/healthz", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def ready():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service is starting.")
    try:
        await run_in_threadpool(app.state.qdrant_client.collection_exists, os.getenv("COLLECTION_NAME", "client"))
    except Exception as error:
        raise HTTPException(status_code=503, detail="Qdrant is unavailable.") from error
    return {"status": "ready"}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_authorize)],
    tags=["chat"],
)
async def chat(request: ChatRequest):
    if sum(len(t.question) + len(t.answer) for t in request.history) > MAX_HISTORY_CHARS:
        raise HTTPException(status_code=422, detail="Conversation history is too long.")
    token_id = (os.getenv("API_ACCESS_TOKEN") and "token") or "client"
    now = time.monotonic()
    with _rate_lock:
        recent = [t for t in _rate_hits.get(token_id, []) if now - t < 60]
        if len(recent) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        recent.append(now)
        _rate_hits[token_id] = recent
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    memory = ConversationMemory(
        turns=[
            ConversationTurn(
                question=turn.question,
                rewritten_question=turn.rewritten_question,
                answer=turn.answer,
            )
            for turn in request.history
        ]
    )
    try:
        answer, sources = await run_in_threadpool(
            answer_question,
            app.state.llm_client,
            app.state.embedding_model,
            app.state.qdrant_client,
            request.query.strip(),
            memory,
        )
    except Exception as error:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502,
            detail="The chat request could not be completed.",
        ) from error

    history = [
        TurnPayload(
            question=turn.question,
            rewritten_question=turn.rewritten_question,
            answer=turn.answer,
        )
        for turn in memory.recent()
    ]
    refused = not sources
    logger.info("chat_completed request_id=%s latency_ms=%.1f refused=%s", request_id, (time.perf_counter()-started)*1000, refused)
    return ChatResponse(answer=answer, sources=sources, refused=refused, history=history)
