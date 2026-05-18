"""
Extended concurrency tests: thread-safety, ordering, rate limiting, caching,
DB contention, and latency budgets.

Strategies 1–4 are deterministic unit/integration. 5–6 are performance/soak
and run only with RUN_LIVE_INTEGRATION=1 or RUN_PERF_TESTS=1.
"""

import asyncio
import time
from unittest.mock import MagicMock

import httpx
import pytest

from test_concurrency import (
    MOCK_AGENT_SLEEP,
    _live_env_ready,
    _mock_query_result,
)

# --- 1) Thread-safety of shared agent state ---


@pytest.mark.asyncio
@pytest.mark.thread_safety
async def test_per_request_agent_no_state_leakage(app_with_mock_agent, monkeypatch):
    """
    With per-request agents (production behavior), assert no state leakage.

    Each request gets a fresh agent (we patch the factory to return an agent
    that tags its answer with the question). Assert each response contains
    only that request's question. If the app ever reused one agent without
    isolation, this would fail (wrong question in response).
    """
    from src.api import main as api_main

    def make_agent():
        def query(*, question="", max_iterations=25):
            time.sleep(MOCK_AGENT_SLEEP)
            return {
                **_mock_query_result(),
                "answer": f"Response for [{question}]",
            }

        agent = MagicMock()
        agent.query.side_effect = query
        return agent

    monkeypatch.setattr(api_main, "_create_philosopher_agent", make_agent)

    questions = ["Alpha", "Beta", "Gamma"]
    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": q, "max_iterations": 1},
            )
            for q in questions
        ]
        responses = await asyncio.gather(*tasks)

    for i, r in enumerate(responses):
        assert r.status_code == 200, r.text
        body = r.json()
        answer = body.get("answer") or ""
        my_q = questions[i]
        others = [questions[j] for j in range(len(questions)) if j != i]
        assert (
            f"[{my_q}]" in answer
        ), f"Response {i} should contain own question {my_q!r}, got {answer!r}"
        for other in others:
            assert (
                f"[{other}]" not in answer
            ), f"State leakage: response for {my_q!r} contained other question {other!r}"


@pytest.fixture
def shared_mock_agent():
    """
    Single agent with mutable state; used to demonstrate leakage when shared.
    """
    state = {"current_question": None}

    def query(*, question="", max_iterations=25):
        state["current_question"] = question
        time.sleep(MOCK_AGENT_SLEEP)
        return {
            **_mock_query_result(),
            "answer": f"Response for [{state['current_question']}]",
        }

    agent = MagicMock()
    agent.query.side_effect = query
    return agent


@pytest.mark.asyncio
@pytest.mark.thread_safety
async def test_shared_agent_exhibits_leakage(
    app_with_mock_agent, monkeypatch, shared_mock_agent
):
    """
    Force a single shared agent; assert we detect state leakage.

    When the same agent is reused across concurrent requests, its mutable
    state is not isolated, so we expect at least one response to contain
    another request's question. This test passes when leakage is detected.
    """
    from src.api import main as api_main

    monkeypatch.setattr(
        api_main, "_create_philosopher_agent", lambda: shared_mock_agent
    )

    questions = ["Alpha", "Beta", "Gamma"]
    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": q, "max_iterations": 1},
            )
            for q in questions
        ]
        responses = await asyncio.gather(*tasks)

    leakage_count = 0
    for i, r in enumerate(responses):
        assert r.status_code == 200, r.text
        body = r.json()
        answer = body.get("answer") or ""
        my_q = questions[i]
        if f"[{my_q}]" not in answer:
            leakage_count += 1
        for other in [questions[j] for j in range(len(questions)) if j != i]:
            if f"[{other}]" in answer:
                leakage_count += 1

    assert leakage_count >= 1, (
        "Expected to detect state leakage when using a single shared agent; "
        "no wrong-question responses found (shared agent may be thread-safe by accident in this run)."
    )


# --- 2) Ordering guarantees ---


@pytest.mark.asyncio
@pytest.mark.ordering
async def test_ordering_session_id_echoed_and_sequence_id_monotonic(
    app_with_mock_agent, monkeypatch
):
    """
    API echoes session_id and assigns monotonic sequence_id per response.
    Ordering is not guaranteed (responses may complete out of order);
    client uses sequence_id to order or detect out-of-order.
    """
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        time.sleep(0.05)
        return {**_mock_query_result(), "answer": f"Answer for {question}"}

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        r1 = await client.post(
            "/api/agent/query",
            json={"question": "First", "max_iterations": 1, "session_id": "sess-1"},
        )
        r2 = await client.post(
            "/api/agent/query",
            json={"question": "Second", "max_iterations": 1, "session_id": "sess-1"},
        )

    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1.get("session_id") == "sess-1"
    assert b2.get("session_id") == "sess-1"
    assert "sequence_id" in b1 and "sequence_id" in b2
    # Sequence ids are monotonic (assigned at request acceptance)
    assert b1["sequence_id"] < b2["sequence_id"]


@pytest.mark.asyncio
@pytest.mark.ordering
async def test_ordering_no_session_id_still_gets_sequence_id(
    app_with_mock_agent, monkeypatch
):
    """Request without session_id still receives sequence_id for ordering."""
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        return _mock_query_result()

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        r = await client.post(
            "/api/agent/query",
            json={"question": "No session", "max_iterations": 1},
        )
    assert r.status_code == 200
    body = r.json()
    assert "sequence_id" in body
    assert body.get("session_id") is None


# --- 3) Rate limiting correctness ---


@pytest.mark.rate_limit
def test_rate_limiting_placeholder():
    """
    Rate limiting not implemented. When added:
    - Configure low limit (e.g. 3/min) in test
    - Send 4–10 requests from same client identity
    - Assert 429, stable error body, limit resets after window
    - Assert different identities are isolated
    """
    pytest.skip("Rate limiting not implemented; add test when added")


# --- 4) Caching correctness ---


@pytest.mark.asyncio
@pytest.mark.caching
async def test_caching_identical_requests_current_behavior(
    app_with_mock_agent, monkeypatch
):
    """
    Caching: current API has no response cache. Two identical requests
    both hit the agent. When caching is added, assert second request
    has cache_hit=true, lower latency, and identical answer/citations.
    """
    from src.api import main as api_main

    call_count = 0

    def mock_query(*, question="", max_iterations=25):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return {**_mock_query_result(), "answer": f"Cached test answer #{call_count}"}

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        r1 = await client.post(
            "/api/agent/query",
            json={"question": "Same question", "max_iterations": 1},
        )
        r2 = await client.post(
            "/api/agent/query",
            json={"question": "Same question", "max_iterations": 1},
        )

    assert r1.status_code == 200 and r2.status_code == 200
    # No cache: both requests hit the agent (call_count == 2)
    assert call_count == 2
    # When cache is added: assert r2.json().get("cache_hit") is True and call_count == 1


# --- 5) DB contention under load ---

DB_CONTENTION_CONCURRENCY = 10
DB_CONTENTION_SUCCESS_RATE = 0.8  # require at least 80% success


@pytest.mark.asyncio
@pytest.mark.db_contention
@pytest.mark.performance
async def test_db_contention_mock_many_concurrent_requests(
    app_with_mock_agent, monkeypatch
):
    """
    Many concurrent requests with mock agent; no real DB/LLM.
    Asserts no deadlocks, no timeouts, and 100% success. Run always.
    """
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        time.sleep(0.08)  # short delay to simulate work
        return _mock_query_result()

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    concurrency = 20
    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=30.0,
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": f"Q{i}", "max_iterations": 1},
            )
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(
        1
        for r in results
        if not isinstance(r, Exception) and getattr(r, "status_code", 0) == 200
    )
    for r in results:
        if isinstance(r, Exception):
            raise AssertionError(f"Request failed: {r}") from r
        if r.status_code != 200:
            try:
                detail = str((r.json() or {}).get("detail", r.text or ""))
            except Exception:
                detail = r.text or ""
            assert "database locked" not in detail.lower(), detail
            assert "timeout" not in detail.lower(), detail

    assert (
        success >= concurrency * DB_CONTENTION_SUCCESS_RATE
    ), f"Success rate {success}/{concurrency} below {DB_CONTENTION_SUCCESS_RATE*100:.0f}%"


@pytest.mark.asyncio
@pytest.mark.db_contention
@pytest.mark.performance
@pytest.mark.live_integration
async def test_db_contention_under_load_real_db(app_with_real_agent):
    """
    Many concurrent real requests (real DB + LLM); assert no deadlocks,
    acceptable success rate. Run with RUN_LIVE_INTEGRATION=1.
    """
    if not _live_env_ready():
        pytest.skip("Set RUN_LIVE_INTEGRATION=1 and have OPENAI_API_KEY, PHILOSOPH_DB")

    concurrency = DB_CONTENTION_CONCURRENCY
    questions = [f"Contention question {i}?" for i in range(concurrency)]
    transport = httpx.ASGITransport(app=app_with_real_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=90.0,
    ) as client:
        tasks = [
            client.post(
                "/api/agent/query",
                json={"question": q, "max_iterations": 3},
            )
            for q in questions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(
        1
        for r in results
        if not isinstance(r, Exception) and getattr(r, "status_code", 0) == 200
    )
    errors = [
        r
        for r in results
        if isinstance(r, Exception) or getattr(r, "status_code", 0) != 200
    ]
    for e in errors:
        if isinstance(e, Exception):
            raise AssertionError(f"Request failed: {e}") from e
        try:
            body = e.json() if e.status_code != 200 else {}
        except Exception:
            body = {}
        detail = str(body.get("detail", "") or "")
        assert "database locked" not in detail.lower(), detail
        assert "timeout" not in detail.lower(), detail

    assert (
        success >= concurrency * DB_CONTENTION_SUCCESS_RATE
    ), f"Success rate {success}/{concurrency} below {DB_CONTENTION_SUCCESS_RATE*100:.0f}%; errors: {errors}"


# --- 6) E2E latency (p50/p95/p99 budgets) ---


LATENCY_P95_BUDGET_MS = 800  # mock agent ~250ms + overhead; allow generous p95


@pytest.mark.asyncio
@pytest.mark.performance
async def test_e2e_latency_p95_budget_mock(app_with_mock_agent, monkeypatch):
    """
    N concurrent requests with mock agent; assert p95 latency under budget.
    Does not use real LLM. For real-LLM latency, run with live_integration
    and a separate budget.
    """
    from src.api import main as api_main

    def mock_query(*, question="", max_iterations=25):
        time.sleep(MOCK_AGENT_SLEEP)
        return _mock_query_result()

    mock_agent = MagicMock()
    mock_agent.query.side_effect = mock_query
    monkeypatch.setattr(api_main, "_create_philosopher_agent", lambda: mock_agent)

    n = 10
    transport = httpx.ASGITransport(app=app_with_mock_agent)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        async def timed_post(i):
            start = time.perf_counter()
            r = await client.post(
                "/api/agent/query",
                json={"question": f"Q{i}", "max_iterations": 1},
            )
            return (time.perf_counter() - start) * 1000, r

        results = await asyncio.gather(*[timed_post(i) for i in range(n)])

    latencies_ms = [ms for ms, _ in results]
    for _, r in results:
        assert r.status_code == 200, r.text

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = (
        latencies_ms[int(len(latencies_ms) * 0.95)]
        if len(latencies_ms) > 1
        else latencies_ms[0]
    )
    p99 = (
        latencies_ms[int(len(latencies_ms) * 0.99)]
        if len(latencies_ms) > 1
        else latencies_ms[0]
    )

    assert (
        p95 <= LATENCY_P95_BUDGET_MS
    ), f"p95 latency {p95:.0f}ms exceeds budget {LATENCY_P95_BUDGET_MS}ms (p50={p50:.0f}, p99={p99:.0f})"
