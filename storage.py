"""
storage.py - persistent storage for multiple chat conversations in a local
SQLite file (conversations.db), so a conversation survives closing the
browser tab, and past conversations can be listed and reopened (like a
sidebar history list).

This module has no dependency on Streamlit - it's plain SQLite, so it can
be reused from app.py (CLI) or any other interface later.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

DB_PATH = "conversations.db"

TITLE_MAX_LENGTH = 60


@dataclass
class ConversationSummary:
    id: str
    title: str
    created_at: str


@dataclass
class StoredMessage:
    role: str  # "user" or "assistant"
    content: str
    sources: list[str]


@contextmanager
def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the tables if they don't exist yet. Safe to call every startup."""
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """)


def create_conversation() -> str:
    """Create a new, untitled conversation and return its id."""
    conversation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conversation_id, "New conversation", created_at),
        )
    return conversation_id


def set_title_from_first_question(conversation_id: str, question: str) -> None:
    """Set the conversation's title from its first question, truncated.
    Only called once, right after the first message of a conversation."""
    title = question.strip().replace("\n", " ")
    if len(title) > TITLE_MAX_LENGTH:
        title = title[:TITLE_MAX_LENGTH].rstrip() + "..."
    with _connect() as connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )


def list_conversations() -> list[ConversationSummary]:
    """Most recent conversation first."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [
        ConversationSummary(row["id"], row["title"], row["created_at"]) for row in rows
    ]


def save_message(
    conversation_id: str, role: str, content: str, sources: list[str]
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, ",".join(sources), created_at),
        )


def load_messages(conversation_id: str) -> list[StoredMessage]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [
        StoredMessage(
            role=row["role"],
            content=row["content"],
            sources=[s for s in row["sources"].split(",") if s],
        )
        for row in rows
    ]


def delete_conversation(conversation_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
