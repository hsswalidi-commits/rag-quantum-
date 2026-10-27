"""Small Cloud Logging compatible structured log helper."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_var: ContextVar[str | None] = ContextVar("cloud_trace", default=None)


def log_event(event: str, *, severity: str = "INFO", **fields: Any) -> None:
    """Emit JSON without user content or credentials."""
    payload: dict[str, Any] = {
        "severity": severity,
        "event": event,
        "request_id": request_id_var.get(),
        **fields,
    }
    trace = trace_var.get()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if trace and project_id:
        payload["logging.googleapis.com/trace"] = (
            f"projects/{project_id}/traces/{trace.split('/')[0]}"
        )
    logging.getLogger("rag").log(
        getattr(logging, severity.upper(), logging.INFO),
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
