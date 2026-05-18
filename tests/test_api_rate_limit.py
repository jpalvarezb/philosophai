"""Tests for rate limiting on agent/query endpoints."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def noop_lifespan():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        yield

    return _noop


@pytest.fixture
def app_ready(monkeypatch, noop_lifespan):
    from src.api import main as api_main
    from unittest.mock import MagicMock

    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)
    app = api_main.create_app()
    api_main.state.ready = True
    api_main.state.philosopher_agent = MagicMock()
    return app, api_main


def test_rate_limit_returns_429_when_over_limit(monkeypatch, app_ready):
    """When record_request returns False, endpoint returns 429 with Retry-After."""
    app, api_main = app_ready

    # Patch where the dependency looks: main imports these from rate_limit
    monkeypatch.setattr(api_main, "record_request", lambda key: False)
    monkeypatch.setattr(api_main, "get_retry_after_seconds", lambda key: 30)

    client = TestClient(app)
    r = client.post(
        "/api/agent/query",
        json={"question": "Hi", "conversation_id": "c1"},
    )

    assert r.status_code == 429
    assert "rate limit" in r.json().get("detail", "").lower()
    assert r.headers.get("Retry-After") == "30"


def test_rate_limit_allows_when_under_limit(monkeypatch, app_ready):
    """When record_request returns True, request proceeds."""
    from unittest.mock import MagicMock
    from src.api import main as api_main

    app, _ = app_ready
    mock_agent = MagicMock()
    mock_agent.query.return_value = {
        "answer": "ok",
        "citations": [],
        "phase": "done",
        "scope": None,
        "collected_chunks": [],
        "collected_entities": [],
        "traversal": {"visited_nodes": [], "edges": [], "edges_traversed": 0},
        "thoughts": [],
        "iterations": 1,
        "session_continued": False,
    }

    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)
    monkeypatch.setattr(api_main, "record_request", lambda key: True)

    client = TestClient(app)
    r = client.post("/api/agent/query", json={"question": "Hi"})

    assert r.status_code == 200
    assert r.json().get("answer") == "ok"
