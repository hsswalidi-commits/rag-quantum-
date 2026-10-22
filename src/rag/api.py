from __future__ import annotations

import logging
import os
import secrets
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
    ingest_knowledge_base,
    load_clients,
)

logger = logging.getLogger(__name__)


class TurnPayload(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    rewritten_question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    history: list[TurnPayload] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    history: list[TurnPayload]


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
        await run_in_threadpool(ingest_knowledge_base, qdrant_client, embedding_model)
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


app = FastAPI(
    title="Quantum Knowledge Assistant API",
    version="1.0.0",
    description="Arabic and English retrieval-augmented question answering API.",
    lifespan=lifespan,
)


@app.get("/healthz", tags=["health"])
def health():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service is starting.")
    return {"status": "ok"}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_authorize)],
    tags=["chat"],
)
async def chat(request: ChatRequest):
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
    return ChatResponse(answer=answer, sources=sources, history=history)
