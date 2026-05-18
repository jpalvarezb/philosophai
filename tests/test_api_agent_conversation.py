"""Tests for Option B: server-side conversation sessions keyed by conversation_id."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient


def _mock_agent_result(answer: str = "Mock answer", session_continued: bool = False):
    return {
        "answer": answer,
        "citations": [],
        "phase": "done",
        "scope": None,
        "collected_chunks": [],
        "collected_entities": [],
        "traversal": {"visited_nodes": [], "edges": [], "edges_traversed": 0},
        "thoughts": [],
        "iterations": 1,
        "session_continued": session_continued,
    }


@pytest.fixture
def noop_lifespan():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        yield

    return _noop


@pytest.fixture
def app_with_mocked_agents(monkeypatch, noop_lifespan):
    """Create app with state.ready and mocked agent creation (no DB/OpenAI)."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)
    app = api_main.create_app()

    api_main.state.ready = True
    api_main.state.philosopher_agent = MagicMock()

    return app, api_main


def test_agent_query_without_conversation_id_uses_fresh_agent(
    monkeypatch, app_with_mocked_agents
):
    """Without conversation_id, each request uses _create_philosopher_agent (new agent)."""
    app, api_main = app_with_mocked_agents

    mock_agent = MagicMock()
    mock_agent.query.return_value = _mock_agent_result("Fresh agent answer")

    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    client = TestClient(app)
    r = client.post(
        "/api/agent/query",
        json={"question": "What is virtue?", "max_iterations": 2},
    )

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Fresh agent answer"
    assert "sequence_id" in data
    mock_agent.query.assert_called_once_with(
        question="What is virtue?", max_iterations=2
    )


def test_agent_query_with_conversation_id_uses_stored_agent(
    monkeypatch, app_with_mocked_agents
):
    """With conversation_id, request uses get_agent_for_conversation and same agent."""
    app, api_main = app_with_mocked_agents

    mock_agent = MagicMock()
    mock_agent.query.return_value = _mock_agent_result(
        "Stored agent answer", session_continued=True
    )
    lock = threading.Lock()

    def fake_get_agent(cid):
        assert cid == "conv-123"
        return mock_agent, lock

    monkeypatch.setattr(api_main, "get_agent_for_conversation", fake_get_agent)

    client = TestClient(app)
    r = client.post(
        "/api/agent/query",
        json={
            "question": "What is virtue?",
            "max_iterations": 2,
            "conversation_id": "conv-123",
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Stored agent answer"
    assert data["session_continued"] is True
    mock_agent.query.assert_called_once_with(
        question="What is virtue?", max_iterations=2
    )


def test_agent_reset_without_body_resets_shared_agent(app_with_mocked_agents):
    """POST /api/agent/reset with no body resets state.philosopher_agent."""
    app, api_main = app_with_mocked_agents

    client = TestClient(app)
    r = client.post("/api/agent/reset")

    assert r.status_code == 200
    data = r.json()
    assert data["message"] == "Session reset"
    assert "conversation_id" not in data
    api_main.state.philosopher_agent.reset_session.assert_called_once()


def test_agent_reset_with_conversation_id_resets_that_agent(
    monkeypatch, app_with_mocked_agents
):
    """POST /api/agent/reset with conversation_id resets that conversation's agent."""
    app, api_main = app_with_mocked_agents

    mock_agent = MagicMock()
    lock = threading.Lock()

    def fake_get_agent(cid):
        assert cid == "conv-456"
        return mock_agent, lock

    monkeypatch.setattr(api_main, "get_agent_for_conversation", fake_get_agent)

    client = TestClient(app)
    r = client.post(
        "/api/agent/reset",
        json={"conversation_id": "conv-456"},
    )

    assert r.status_code == 200
    data = r.json()
    assert data["message"] == "Conversation reset"
    assert data["conversation_id"] == "conv-456"
    mock_agent.reset_session.assert_called_once()
    api_main.state.philosopher_agent.reset_session.assert_not_called()
