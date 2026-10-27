from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import JSONResponse, Response

from rag.core import (
    MAX_HISTORY_TURNS,
    ConversationMemory,
    ConversationTurn,
    answer_question,
    load_clients,
)
from rag.observability import log_event, request_id_var, trace_var

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


MAX_HISTORY_CHARS = _positive_int("MAX_HISTORY_CHARS", 12000)
RATE_LIMIT_PER_MINUTE = _positive_int("RATE_LIMIT_PER_MINUTE", 20)
MAX_BODY_BYTES = _positive_int("MAX_BODY_BYTES", 65536)
_RATE_LOCK = threading.Lock()
_RATE_HITS: dict[str, deque[float]] = defaultdict(deque)


class TurnPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    rewritten_question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=3000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    history: list[TurnPayload] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    history: list[TurnPayload]


def _authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    token = os.getenv("API_ACCESS_TOKEN")
    # Local development can opt out; a deployed Cloud Run revision never can.
    if not token and not os.getenv("K_SERVICE"):
        return
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API access is not configured.",
        )
    scheme, _, supplied_token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied_token or not secrets.compare_digest(supplied_token, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.ready = False
    try:
        llm_client, embedding_model, qdrant_client = await run_in_threadpool(load_clients)
        application.state.llm_client = llm_client
        application.state.embedding_model = embedding_model
        application.state.qdrant_client = qdrant_client
        application.state.ready = True
        log_event("application_started", model=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    except Exception as error:
        log_event("application_start_failed", severity="ERROR", error_type=type(error).__name__)
        raise
    try:
        yield
    finally:
        application.state.ready = False
        close = getattr(qdrant_client, "close", None)
        if close:
            await run_in_threadpool(close)
        close = getattr(llm_client, "close", None)
        if close:
            await run_in_threadpool(close)


docs_enabled = os.getenv("ENABLE_DOCS", "0").strip().lower() in {"1", "true", "yes"}
app = FastAPI(
    title="Quantum Knowledge Assistant API",
    version="1.0.0",
    description="Arabic and English retrieval-augmented question answering API.",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_enabled else None,
)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        max_age=600,
    )


@app.middleware("http")
async def request_observability(request: Request, call_next) -> Response:
    raw_request_id = request.headers.get("X-Request-ID", "")
    request_id = raw_request_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", raw_request_id) else str(uuid.uuid4())
    request_token = request_id_var.set(request_id)
    trace_header = request.headers.get("X-Cloud-Trace-Context", "")
    trace_token = trace_var.set(trace_header.split("/", 1)[0] if trace_header else None)
    started = time.perf_counter()
    response: Response
    try:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                parsed_content_length = int(content_length)
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
            else:
                if parsed_content_length < 0:
                    response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
                elif parsed_content_length > MAX_BODY_BYTES:
                    response = JSONResponse(status_code=413, content={"detail": "Request body too large."})
                else:
                    response = await call_next(request)
        else:
            response = await call_next(request)
    except Exception as error:  # noqa: BLE001 - ASGI boundary must return a generic error
        log_event("request_failed", severity="ERROR", error_type=type(error).__name__)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error."})
    response.headers["X-Request-ID"] = request_id
    log_event(
        "http_request_complete",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    trace_var.reset(trace_token)
    request_id_var.reset(request_token)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _error: RequestValidationError) -> JSONResponse:
    # Do not echo user payloads or Pydantic's internal error context to clients.
    return JSONResponse(status_code=422, content={"detail": "Invalid request."})


def _rate_limit(request: Request) -> None:
    remote_ip = request.client.host if request.client else "unknown"
    key = hashlib.sha256(remote_ip.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _RATE_LOCK:
        hits = _RATE_HITS[key]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        # Bound memory use if clients vary source addresses aggressively.
        if len(_RATE_HITS) > 10000:
            for stale_key, stale_hits in list(_RATE_HITS.items()):
                if not stale_hits or now - stale_hits[-1] >= 60:
                    _RATE_HITS.pop(stale_key, None)


@app.get("/healthz", tags=["health"])
def health() -> dict[str, str]:
    # Liveness must stay cheap and independent of Qdrant/OpenRouter.
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def ready() -> dict[str, str]:
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service is starting.")
    try:
        exists = await run_in_threadpool(
            app.state.qdrant_client.collection_exists,
            os.getenv("COLLECTION_NAME", "client"),
        )
        if not exists:
            log_event("readiness_collection_missing", severity="WARNING")
            raise HTTPException(status_code=503, detail="Knowledge collection is unavailable.")
    except HTTPException:
        raise
    except Exception as error:
        log_event("readiness_check_failed", severity="WARNING", error_type=type(error).__name__)
        raise HTTPException(status_code=503, detail="Qdrant is unavailable.") from error
    return {"status": "ready"}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_authorize)],
    tags=["chat"],
)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    history_chars = sum(
        len(turn.question) + len(turn.rewritten_question) + len(turn.answer)
        for turn in request.history
    )
    if history_chars > MAX_HISTORY_CHARS:
        raise HTTPException(status_code=422, detail="Conversation history is too long.")
    _rate_limit(http_request)

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
            request.query,
            memory,
        )
    except Exception as error:
        log_event("chat_request_failed", severity="ERROR", error_type=type(error).__name__)
        raise HTTPException(
            status_code=502,
            detail="The chat request could not be completed.",
        ) from error

    response_history = [
        TurnPayload(
            question=turn.question,
            rewritten_question=turn.rewritten_question,
            answer=turn.answer,
        )
        for turn in memory.recent()
    ]
    refused = not sources
    log_event("chat_completed", refused=refused, source_count=len(sources))
    return ChatResponse(
        answer=answer,
        sources=sources,
        refused=refused,
        history=response_history,
    )
