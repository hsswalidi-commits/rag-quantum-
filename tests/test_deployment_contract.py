from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag.api import ChatRequest, _authorize, app

ROOT = Path(__file__).resolve().parents[1]


def test_health_and_chat_routes_are_registered() -> None:
    routes = {route.path for route in app.routes}

    assert {"/healthz", "/readyz", "/v1/chat"}.issubset(routes)


def test_chat_requires_the_configured_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "rag-agent-app")
    monkeypatch.setenv("API_ACCESS_TOKEN", "unit-test-token")

    with pytest.raises(HTTPException) as error:
        _authorize("Bearer invalid-token")

    assert error.value.status_code == 401


def test_configured_bearer_token_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "rag-agent-app")
    monkeypatch.setenv("API_ACCESS_TOKEN", "unit-test-token")

    assert _authorize("Bearer unit-test-token") is None


def test_chat_request_rejects_blank_queries() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query="   ")


def test_terraform_deployment_contract() -> None:
    terraform = (ROOT / "terraform" / "main.tf").read_text(encoding="utf-8")

    for expected in (
        'bucket = "leap-coop-rag-terraform-state"',
        'prefix = "rag-agent/production"',
        'default     = "leap-coop"',
        'default     = "me-central1"',
        'name                = "rag-agent-app"',
        'repository_id = "rag-agent"',
        'member   = "allUsers"',
        'roles/secretmanager.secretAccessor',
        '"rag-openrouter-api-key"',
        '"rag-qdrant-api-key"',
        '"rag-qdrant-url"',
        '"rag-api-bearer-token"',
    ):
        assert expected in terraform


def test_workflow_requires_expected_github_secrets_and_artifact_registry() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    for name in (
        "GCP_SA_KEY",
        "OPENROUTER_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "API_BEARER_TOKEN",
    ):
        assert f"secrets.{name}" in workflow

    assert "${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-agent/${SERVICE}:${GITHUB_SHA}" in workflow
    assert "toJSON(secrets)" not in workflow
