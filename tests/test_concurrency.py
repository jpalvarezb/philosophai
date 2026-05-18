"""
Concurrency tests for the PhilosopherAgent API.

- Mock-based tests: verify the API runs multiple requests in parallel (run fast, no LLM).
- Real-agent test: uses real PhilosopherAgent and real LLM; skipped if OPENAI_API_KEY
  or DB not available. Run with: pytest -m live_integration
"""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

# Simulated LLM latency per request (seconds)
MOCK_AGENT_SLEEP = 0.25

# If requests ran serially, N requests would take at least N * MOCK_AGENT_SLEEP.
# We assert total time is under this to prove concurrency (allow some overhead).
MAX_WALL_TIME_3 = 0.9  # 3 * 0.25 = 0.75; 0.9 allows slack
MAX_WALL_TIME_5 = 1.35  # 5 * 0.25 = 1.25; 1.35 allows slack for 5-way e2e

NUM_CONCURRENT_REQUESTS = 3
NUM_E2E_PARALLEL = 5


def _mock_query_result():
    """Minimal dict matching AgentQueryResponse for the mock agent."""
    return {
        "answer": "Concurrent test answer.",
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


@pytest.fixture
def app_with_mock_agent(monkeypatch, noop_lifespan, reset_api_state):
    """FastAPI app with lifespan disabled and state.ready True."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)
    app = api_main.create_app()
    api_main.state.ready = True
    return app


def _live_env_ready():
    """True if OPENAI_API_KEY and PHILOSOPH_DB exist and RUN_LIVE_INTEGRATION=1."""
    if os.environ.get("RUN_LIVE_INTEGRATION", "").strip() != "1":
        return False
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    from src.api import main as api_main

    db_path_raw = os.environ.get("PHILOSOPH_DB", "data/philosoph.duckdb")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        root = Path(api_main.__file__).resolve().parent.parent.parent
        db_path = (root / db_path_raw).resolve()
    return db_path.exists()


@pytest.fixture
def app_with_real_agent(monkeypatch, noop_lifespan, reset_api_state):
    """
    Real app: init_components() runs (real DB, real OpenAI). No LLM mock.
    Skipped if OPENAI_API_KEY or DB not available.
    """
    from src.api import main as api_main

    if not _live_env_ready():
        pytest.skip(
            "Set RUN_LIVE_INTEGRATION=1 and have OPENAI_API_KEY and PHILOSOPH_DB for real-agent test"
        )

    api_main.init_components()
    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)
    app = api_main.create_app()
    return app


@pytest.mark.asyncio
async def test_agent_query_requests_run_concurrently(app_with_mock_agent, monkeypatch):
    """
    Fire N concurrent POST /api/agent/query requests.

    The endpoint runs each request in run_in_executor with a fresh agent.
    We mock _create_philosopher_agent to return an agent whose .query() sleeps
    then returns. If requests ran in parallel, wall time < N * sleep.
    """
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        time.sleep(MOCK_AGENT_SLEEP)
        return _mock_query_result()

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    start = time.perf_counter()
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": "Test?", "max_iterations": 1},
            )
            for _ in range(NUM_CONCURRENT_REQUESTS)
        ]
        responses = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    assert len(responses) == NUM_CONCURRENT_REQUESTS
    for r in responses:
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("answer") == "Concurrent test answer."

    assert elapsed < MAX_WALL_TIME_3, (
        f"Requests took {elapsed:.2f}s; if parallel expect < {MAX_WALL_TIME_3}s. "
        "Concurrency may not be working."
    )


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_five_e2e_agent_requests_run_in_parallel(
    app_with_mock_agent, monkeypatch
):
    """
    E2E-style: 5 concurrent POST /api/agent/query requests.

    Proves the app handles 5-way concurrency; wall time must be well under
    5 * MOCK_AGENT_SLEEP (serial would be ~1.25s).
    """
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        time.sleep(MOCK_AGENT_SLEEP)
        return _mock_query_result()

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    start = time.perf_counter()
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": f"E2E question {i}?", "max_iterations": 1},
            )
            for i in range(NUM_E2E_PARALLEL)
        ]
        responses = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    assert len(responses) == NUM_E2E_PARALLEL
    for i, r in enumerate(responses):
        assert r.status_code == 200, f"Request {i}: {r.text}"
        body = r.json()
        assert body.get("answer") == "Concurrent test answer."

    assert elapsed < MAX_WALL_TIME_5, (
        f"5 requests took {elapsed:.2f}s; if parallel expect < {MAX_WALL_TIME_5}s. "
        "Five-way concurrency may not be working."
    )


@pytest.mark.asyncio
async def test_greeting_requests_run_concurrently(app_with_mock_agent, monkeypatch):
    """
    Fire N concurrent GET /api/agent/greeting requests.

    Greeting uses a fresh agent and run_in_executor; we mock _create_philosopher_agent
    so the agent's generate_greeting() sleeps then returns. Wall time should be
    under N * sleep if parallel.
    """
    from src.api import main as api_main

    def mock_greeting():
        time.sleep(MOCK_AGENT_SLEEP)
        return "Philo here—what would you like to explore?"

    mock_agent = MagicMock()
    mock_agent.generate_greeting.side_effect = mock_greeting
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    start = time.perf_counter()
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        tasks = [
            client.get("/api/agent/greeting") for _ in range(NUM_CONCURRENT_REQUESTS)
        ]
        responses = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    assert len(responses) == NUM_CONCURRENT_REQUESTS
    for r in responses:
        assert r.status_code == 200, r.text
        assert "greeting" in r.json()

    assert (
        elapsed < MAX_WALL_TIME_3
    ), f"Greeting requests took {elapsed:.2f}s; expect < {MAX_WALL_TIME_3}s if parallel."


@pytest.mark.asyncio
@pytest.mark.live_integration
async def test_real_philosopher_agent_concurrent_queries(app_with_real_agent):
    """
    Real PhilosopherAgent and real LLM: 3 concurrent queries.

    No mocks. Each request gets a fresh agent via _create_philosopher_agent.
    Asserts all succeed and each response has an answer (no cross-talk).
    Skipped unless RUN_LIVE_INTEGRATION=1, OPENAI_API_KEY, and PHILOSOPH_DB exist.
    Slow (real API calls). Run explicitly: RUN_LIVE_INTEGRATION=1 pytest -m live_integration -v
    """
    questions = [
        "What is wisdom?",
        "What is virtue?",
        "What is knowledge?",
    ]
    transport = httpx.ASGITransport(app=app_with_real_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=120.0,
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": q, "max_iterations": 10},
            )
            for q in questions
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            raise AssertionError(f"Request {i} ({questions[i]!r}) failed: {r}") from r
        assert r.status_code == 200, f"Request {i}: {r.status_code} {r.text}"
        body = r.json()
        assert "answer" in body, body
        assert (
            len((body.get("answer") or "").strip()) > 0
        ), f"Empty answer for {questions[i]!r}"
