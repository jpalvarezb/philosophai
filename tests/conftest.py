import pytest


@pytest.fixture
def noop_lifespan():
    """A lifespan context manager that does nothing.

    Useful to create a FastAPI app in tests without initializing external dependencies
    (DB, OpenAI client, etc.).
    """

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        yield

    return _noop


@pytest.fixture
def reset_api_state():
    """Reset global API state after each test."""

    from src.api import main as api_main

    old = {
        "ready": api_main.state.ready,
        "storage": api_main.state.storage,
        "graph_builder": api_main.state.graph_builder,
        "agent": api_main.state.agent,
        "philosopher_agent": api_main.state.philosopher_agent,
        "agent_tools": api_main.state.agent_tools,
        "citation_builder": api_main.state.citation_builder,
        "node_to_community": dict(api_main.state.node_to_community or {}),
        "openai_client": api_main.state.openai_client,
        "vector_search": getattr(api_main.state, "vector_search", None),
    }

    try:
        yield api_main.state
    finally:
        api_main.state.ready = old["ready"]
        api_main.state.storage = old["storage"]
        api_main.state.graph_builder = old["graph_builder"]
        api_main.state.agent = old["agent"]
        api_main.state.philosopher_agent = old["philosopher_agent"]
        api_main.state.agent_tools = old["agent_tools"]
        api_main.state.citation_builder = old["citation_builder"]
        api_main.state.node_to_community = old["node_to_community"]
        api_main.state.openai_client = old["openai_client"]
        if "vector_search" in old:
            api_main.state.vector_search = old["vector_search"]
