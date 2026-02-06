import networkx as nx
import pytest
from starlette.testclient import TestClient


def test_api_graph_returns_string_ids(monkeypatch, noop_lifespan, reset_api_state):
    from src.api import main as api_main

    # Disable real startup initialization
    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)

    app = api_main.create_app()

    # Provide a minimal graph
    G = nx.MultiDiGraph()
    # Use non-string node id to ensure the endpoint coerces it
    G.add_node(123, label=None)
    G.add_node("bond of union", label="bond of union")
    G.add_edge(123, "bond of union", key="serves_as", label=None, weight=2)

    api_main.state.graph_builder = type("GB", (), {"graph": G})()
    api_main.state.ready = True

    # Int key mapping should still work due to endpoint fallback
    api_main.state.node_to_community = {123: 9, "bond of union": 1}

    client = TestClient(app)
    r = client.get("/api/graph?limit=1000&main_only=false")
    assert r.status_code == 200

    payload = r.json()
    assert isinstance(payload, dict)
    assert "nodes" in payload and "links" in payload

    nodes = payload["nodes"]
    links = payload["links"]
    assert isinstance(nodes, list)
    assert isinstance(links, list)

    # Nodes must be objects, never primitives
    assert all(isinstance(n, dict) for n in nodes)
    assert all(isinstance(l, dict) for l in links)

    # IDs must be strings
    for n in nodes:
        assert isinstance(n.get("id"), str)
        assert n.get("id")
        assert isinstance(n.get("label"), str)

    for l in links:
        assert isinstance(l.get("source"), str)
        assert isinstance(l.get("target"), str)
        assert l.get("source")
        assert l.get("target")


@pytest.mark.parametrize("limit", [1, 2, 100])
def test_api_graph_limit_is_respected(monkeypatch, noop_lifespan, reset_api_state, limit):
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "lifespan", noop_lifespan)
    app = api_main.create_app()

    G = nx.MultiDiGraph()
    # Create a connected component
    for i in range(5):
        G.add_node(str(i), label=str(i))
    for i in range(4):
        G.add_edge(str(i), str(i + 1), key="k", label="rel", weight=1)

    api_main.state.graph_builder = type("GB", (), {"graph": G})()
    api_main.state.ready = True

    client = TestClient(app)
    r = client.get(f"/api/graph?limit={limit}&main_only=true")
    assert r.status_code == 200
    payload = r.json()

    nodes = payload["nodes"]
    # limit applies after degree ordering; when limit < node count, should cap
    assert len(nodes) <= limit if limit > 0 else True
