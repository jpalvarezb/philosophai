"""FastAPI application for PhilosophAI with GraphRAG."""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

# Environment: "development" (local) vs "production" (Fly/cloud).
# Set PHILOSOPH_ENV explicitly; if unset, we treat Fly (FLY_APP_NAME) as production, else development.
def _env_mode() -> Literal["development", "production"]:
    env = os.environ.get("PHILOSOPH_ENV", "").lower()
    if env in ("dev", "development", "local"):
        return "development"
    if env in ("prod", "production") or os.environ.get("FLY_APP_NAME"):
        return "production"
    return "development"
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .rate_limit import get_client_key, get_retry_after_seconds, record_request
from .ws import (
    router as ws_router,
    set_agent,
    set_get_agent_for_conversation,
    set_philosopher_agent,
    set_philosopher_agent_factory,
)


# --- Pydantic Models ---
class ScopeRequest(BaseModel):
    authors: list[str] = []
    titles: list[str] = []
    traditions: list[str] = []
    domains: list[str] = []
    strict: bool = True


class QueryRequest(BaseModel):
    question: str
    max_hops: int = 3
    max_context_chunks: int = 12
    use_community_routing: bool = True
    scope: ScopeRequest | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    traversal: dict
    trace: dict


# --- Pydantic Models for Agentic Query ---
class AgentQueryRequest(BaseModel):
    question: str
    max_iterations: int = 25
    verbose: bool = False
    session_id: str | None = None  # Optional; echoed in response for ordering/tracking
    conversation_id: str | None = None  # Optional; when set, reuses server-side agent (last 5 Q/As) for this conversation


class AgentQueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    phase: str
    scope: dict | None
    collected_chunks: list[str]
    collected_entities: list[str]
    traversal: dict
    thoughts: list[dict]
    iterations: int
    session_continued: bool = False
    session_id: str | None = None  # Echo of request session_id, if provided
    sequence_id: int = 0  # Monotonic id per response; client can use to order or detect out-of-order


class AgentResetRequest(BaseModel):
    conversation_id: str | None = None  # If set, reset only this conversation's agent; else reset shared agent (legacy)


# --- Application State ---
class AppState:
    """Holds initialized components."""
    def __init__(self):
        self.storage = None
        self.graph_builder = None
        self.agent = None
        self.philosopher_agent = None  # Agentic query handler
        self.agent_tools = None
        self.citation_builder = None
        self.node_to_community = {}
        self.openai_client = None
        self.vector_search = None  # For per-request PhilosopherAgent creation
        self.ready = False


state = AppState()

# Lock for any use of the shared philosopher_agent (e.g. reset_session)
_shared_philosopher_agent_lock = threading.Lock()

# Monotonic sequence for agent query responses (ordering / out-of-order detection)
_query_sequence_lock = threading.Lock()
_query_sequence_counter = 0

# Option B: server-side sessions keyed by conversation_id
_CONVERSATION_TTL_SECONDS = int(os.environ.get("PHILOSOPH_CONVERSATION_TTL", "1800"))  # 30 min
_CONVERSATION_MAX_ENTRIES = int(os.environ.get("PHILOSOPH_CONVERSATION_MAX", "200"))
_conversation_store: dict[str, tuple[Any, float]] = {}  # conversation_id -> (agent, last_used_ts)
_conversation_locks: dict[str, threading.Lock] = {}
_store_lock = threading.Lock()


def _evict_conversations():
    """Remove expired entries and trim to max size. Caller must hold _store_lock."""
    now = time.monotonic()
    to_remove = [
        cid for cid, (_, last) in _conversation_store.items()
        if now - last > _CONVERSATION_TTL_SECONDS
    ]
    if len(_conversation_store) > _CONVERSATION_MAX_ENTRIES:
        by_age = sorted(_conversation_store.items(), key=lambda x: x[1][1])
        n = len(_conversation_store) - _CONVERSATION_MAX_ENTRIES
        for cid, _ in by_age[:n]:
            to_remove.append(cid)
    for cid in to_remove:
        _conversation_store.pop(cid, None)
        _conversation_locks.pop(cid, None)


def get_agent_for_conversation(conversation_id: str):
    """
    Return (agent, lock) for the given conversation_id.
    Caller must hold the returned lock while using the agent (e.g. running a query).
    """
    with _store_lock:
        _evict_conversations()
        now = time.monotonic()
        if conversation_id in _conversation_store:
            agent, _ = _conversation_store[conversation_id]
            _conversation_store[conversation_id] = (agent, now)
            return agent, _conversation_locks[conversation_id]
        agent = _create_philosopher_agent()
        lock = threading.Lock()
        _conversation_store[conversation_id] = (agent, now)
        _conversation_locks[conversation_id] = lock
        return agent, lock


def _rate_limit_dep(request: Request):
    """Dependency: raise 429 if client is over rate limit."""
    key = get_client_key(request)
    if not record_request(key):
        retry = get_retry_after_seconds(key)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry)} if retry else {},
        )


def _create_philosopher_agent():
    """Create a new PhilosopherAgent with its own AgentTools for concurrent, isolated queries."""
    from ..agents import AgentTools, PhilosopherAgent

    tools = AgentTools(
        state.storage,
        state.graph_builder,
        state.vector_search,
        state.node_to_community,
    )
    return PhilosopherAgent(
        agent_tools=tools,
        citation_builder=state.citation_builder,
        llm_client=state.openai_client,
        llm_model="gpt-4o",
        verbose=False,
    )


def init_components():
    """Initialize all components (called at startup)."""
    from ..config import setup_logging
    setup_logging()

    from openai import OpenAI
    from ..storage import DuckDBStorage
    from ..graph import GraphBuilder, GraphTraverser, GraphFilters
    from ..rag import VectorSearch, ResultFusion, CitationBuilder
    from ..agents import MultiHopAgent, AgentTools, PhilosopherAgent

    # Config from environment
    db_path_raw = os.environ.get("PHILOSOPH_DB", "data/philosoph.duckdb")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        # Resolve relative paths against project root (parent of src/)
        _project_root = Path(__file__).resolve().parent.parent.parent
        db_path = (_project_root / db_path_raw).resolve()
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    print(f"📂 Initializing with DB: {db_path}")

    # Storage
    state.storage = DuckDBStorage(db_path)

    # Graph
    print("🏗️ Building graph...")
    state.graph_builder = GraphBuilder(state.storage)
    G = state.graph_builder.build()

    # Community mappings
    print("📊 Loading community mappings...")
    communities_df = state.storage.get_communities()
    for _, row in communities_df.iterrows():
        for node_id in row["node_ids"]:
            state.node_to_community[node_id] = row["community_id"]

    # Ensure membership table is populated
    state.storage.populate_community_membership()

    # Agent components
    print("🔧 Initializing agent...")
    client = OpenAI(api_key=openai_key)
    vector_search = VectorSearch(state.storage, client)
    state.vector_search = vector_search
    fusion = ResultFusion(state.storage)
    state.citation_builder = CitationBuilder(state.storage, state.node_to_community)

    # Initialize filters for traversal and seeding (uses precomputed conceptness scores)
    filters = GraphFilters(G, storage=state.storage, hub_threshold_pct=0.01, min_degree=1)
    traverser = GraphTraverser(G, state.node_to_community, filters=filters)

    state.agent = MultiHopAgent(
        storage=state.storage,
        graph_builder=state.graph_builder,
        vector_search=vector_search,
        fusion=fusion,
        citation_builder=state.citation_builder,
        traverser=traverser,
        llm_client=client,
        node_to_community=state.node_to_community,
        filters=filters,
    )

    # Initialize AgentTools for the agentic query handler
    state.agent_tools = AgentTools(
        storage=state.storage,
        graph_builder=state.graph_builder,
        vector_search=vector_search,
        node_to_community=state.node_to_community,
    )

    # Initialize PhilosopherAgent (OpenAI function calling-based agentic handler)
    print("🤖 Initializing PhilosopherAgent...")
    state.philosopher_agent = PhilosopherAgent(
        agent_tools=state.agent_tools,
        citation_builder=state.citation_builder,
        llm_client=client,
        llm_model="gpt-4o",
        verbose=False,
    )
    state.openai_client = client

    # Share agents with WebSocket module
    set_agent(state.agent)
    set_philosopher_agent(state.philosopher_agent)
    set_philosopher_agent_factory(_create_philosopher_agent)
    set_get_agent_for_conversation(get_agent_for_conversation)

    state.ready = True
    print("✅ PhilosophAI ready!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    init_components()
    yield
    if state.storage:
        state.storage.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PHILO-001",
        description="Knowledge Graph RAG with Community Routing",
        version="0.8.0",
        lifespan=lifespan,
    )

    # CORS - Environment-specific defaults; override with CORS_ORIGINS
    mode = _env_mode()
    if os.environ.get("CORS_ORIGINS"):
        allowed_origins_env = os.environ.get("CORS_ORIGINS")
    elif mode == "development":
        allowed_origins_env = "http://localhost:8000,http://localhost:5713,http://localhost:3000,http://127.0.0.1:8000,http://127.0.0.1:5713,http://127.0.0.1:3000"
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    print(f"🔒 CORS ({mode}): {allowed_origins}")

    _rlimit = int(os.environ.get("PHILOSOPH_RATE_LIMIT_REQUESTS"))
    _rwin = int(os.environ.get("PHILOSOPH_RATE_LIMIT_WINDOW_SECONDS"))
    if _rlimit <= 0:
        print("⏱️ Rate limit: disabled")
    else:
        print(f"⏱️ Rate limit: {_rlimit} requests / {_rwin}s per client")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Include WebSocket router
    app.include_router(ws_router, prefix="/api")

    # Health check (includes app version for deploy verification)
    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": state.ready, "version": app.version}

    # Query endpoint (non-streaming)
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest, _: None = Depends(_rate_limit_dep)):
        """Execute a GraphRAG query."""
        from ..agents import Scope

        if not state.ready or not state.agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Build scope if provided
        scope = None
        if request.scope:
            scope = Scope(
                authors=request.scope.authors,
                titles=request.scope.titles,
                traditions=request.scope.traditions,
                domains=request.scope.domains,
                strict=request.scope.strict,
            )

        result = state.agent.query(
            question=request.question,
            max_hops=request.max_hops,
            max_context_chunks=request.max_context_chunks,
            use_community_routing=request.use_community_routing,
            scope=scope,
        )
        return result

    # Agentic query endpoint (PhilosopherAgent with sequential thinking)
    @app.post("/api/agent/query", response_model=AgentQueryResponse)
    async def agent_query(request: AgentQueryRequest, _: None = Depends(_rate_limit_dep)):
        """
        Execute an agentic query with sequential thinking.

        If conversation_id is set, reuses the same server-side agent for that conversation
        (enables last-5-Q/A follow-up context). Otherwise creates a fresh agent per request.
        """
        if not state.ready:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        global _query_sequence_counter
        with _query_sequence_lock:
            next_seq = _query_sequence_counter
            _query_sequence_counter += 1

        if request.conversation_id:
            agent, conv_lock = get_agent_for_conversation(request.conversation_id)

            def run_query():
                with conv_lock:
                    return agent.query(
                        question=request.question,
                        max_iterations=request.max_iterations,
                    )
        else:
            def run_query():
                agent = _create_philosopher_agent()
                return agent.query(
                    question=request.question,
                    max_iterations=request.max_iterations,
                )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_query)
            result["sequence_id"] = next_seq
            result["session_id"] = request.session_id
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Agentic greeting (LLM-generated); uses fresh agent so concurrent greetings don't share state
    @app.post("/api/agent/reset")
    async def agent_reset(body: AgentResetRequest | None = Body(None), _: None = Depends(_rate_limit_dep)):
        """Reset conversation state. If body.conversation_id is set, reset that conversation's agent; else reset shared agent (legacy)."""
        if not state.ready or not state.philosopher_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        if body and body.conversation_id:
            agent, conv_lock = get_agent_for_conversation(body.conversation_id)

            def do_reset():
                with conv_lock:
                    agent.reset_session()

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, do_reset)
                return {"status": "ok", "message": "Conversation reset", "conversation_id": body.conversation_id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            def do_reset():
                with _shared_philosopher_agent_lock:
                    state.philosopher_agent.reset_session()

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, do_reset)
                return {"status": "ok", "message": "Session reset"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/agent/greeting")
    async def agent_greeting(_: None = Depends(_rate_limit_dep)):
        if not state.ready:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        def do_greeting():
            agent = _create_philosopher_agent()
            return agent.generate_greeting()

        try:
            loop = asyncio.get_event_loop()
            greeting = await loop.run_in_executor(None, do_greeting)
            return {"greeting": greeting}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Graph data for visualization
    @app.get("/api/graph")
    async def get_graph_data(
        limit: int = 1000,
        main_only: bool = True,
    ):
        """
        Return graph nodes/edges for visualization.

        - main_only=True (default): restrict to the largest connected component (main graph)
        - limit: take the top-N highest-degree nodes within that component (or all if limit <= 0)
        """
        if not state.ready:
            raise HTTPException(status_code=503, detail="Not initialized")

        import networkx as nx

        G = state.graph_builder.graph

        # Choose the subgraph (largest connected component when requested)
        if main_only:
            undirected = G.to_undirected()
            try:
                largest_nodes = max(nx.connected_components(undirected), key=len)
            except ValueError:
                largest_nodes = set()
            H = G.subgraph(largest_nodes)
        else:
            H = G

        # Degree-based ordering within the chosen subgraph
        deg_map = dict(H.degree())
        ordered = sorted(deg_map.items(), key=lambda x: x[1], reverse=True)
        if limit > 0:
            selected_nodes = [n for n, _ in ordered[:limit]]
        else:
            selected_nodes = list(H.nodes())
        node_set = set(selected_nodes)

        nodes = []
        links = []

        for node_id in node_set:
            data = H.nodes.get(node_id, {})
            node_id_str = str(node_id)
            label = data.get("label", node_id_str)
            nodes.append({
                "id": node_id_str,
                "label": str(label) if label is not None else node_id_str,
                "community": state.node_to_community.get(node_id, state.node_to_community.get(node_id_str)),
                "degree": int(deg_map.get(node_id, 0)),
            })

        for u, v, data in H.edges(data=True):
            if u in node_set and v in node_set:
                label = data.get("label", "")
                links.append({
                    "source": str(u),
                    "target": str(v),
                    "label": str(label) if label is not None else "",
                    "weight": int(data.get("weight", 1) or 1),
                })

        return {"nodes": nodes, "links": links}


    # Communities endpoint
    @app.get("/api/communities")
    async def get_communities():
        """Return all communities with reports."""
        if not state.ready:
            raise HTTPException(status_code=503, detail="Not initialized")

        communities_df = state.storage.get_communities()
        reports_df = state.storage.get_community_reports()

        # Merge reports with communities
        report_map = {}
        if not reports_df.empty:
            for _, row in reports_df.iterrows():
                report_map[row["comm_id"]] = row["report_text"]

        result = []
        for _, row in communities_df.iterrows():
            top_terms = row["top_terms"]
            if top_terms is None or (hasattr(top_terms, '__len__') and len(top_terms) == 0):
                top_terms = []
            else:
                top_terms = list(top_terms)[:10]

            result.append({
                "community_id": int(row["community_id"]),
                "size": int(row["size"]),
                "top_terms": top_terms,
                "summary": row["summary"],
                "report": report_map.get(row["community_id"]),
            })

        return {"communities": result}

    # Single community detail
    @app.get("/api/communities/{comm_id}")
    async def get_community(comm_id: int):
        """Get details for a specific community."""
        if not state.ready:
            raise HTTPException(status_code=503, detail="Not initialized")

        # Get nodes in this community
        nodes = state.storage.get_nodes_in_communities([comm_id])

        # Get community info
        communities_df = state.storage.get_communities()
        row = communities_df[communities_df["community_id"] == comm_id]
        if row.empty:
            raise HTTPException(status_code=404, detail="Community not found")

        row = row.iloc[0]

        top_terms = row["top_terms"]
        if top_terms is None or (hasattr(top_terms, '__len__') and len(top_terms) == 0):
            top_terms = []
        else:
            top_terms = list(top_terms)

        return {
            "community_id": comm_id,
            "size": int(row["size"]),
            "top_terms": top_terms,
            "summary": row["summary"],
            "node_ids": nodes[:100],  # Cap for response size
        }

    # Export artifacts (client sends payload; no conversation storage)
    @app.post("/api/export/transcript", response_class=PlainTextResponse)
    async def export_transcript(payload: dict[str, Any]):
        """Generate TXT transcript with renumbered citations. Body: export payload (messages + graph_trace)."""
        from .export_artifacts import build_transcript_text
        try:
            text = build_transcript_text(payload)
            return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/export/report-html")
    async def export_report_html(payload: dict[str, Any]):
        """Generate interactive HTML report (zoom/pan graph). Body: export payload (messages + graph_trace)."""
        try:
            from .export_artifacts import build_report_html
            html = build_report_html(payload)
            return Response(content=html, media_type="text/html; charset=utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # Graph-only HTML for download (no hosting; client saves as philo_graph.html)
    @app.post("/api/export/field-html")
    async def export_field_html(payload: dict[str, Any]):
        """Return graph-only HTML for client to download as philo_graph.html. Body: same as other export payloads."""
        try:
            from .export_artifacts import build_report_html
            html = build_report_html({**payload, "embed_only": True})
            return {"html": html}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # Serve UI: catch-all route last so /api/* and /health match first (StaticFiles at "/" can take precedence otherwise)
    ui_path = Path(__file__).resolve().parent.parent.parent / "ui"
    if ui_path.exists():
        @app.get("/{full_path:path}")
        async def serve_ui(full_path: str):
            # Only serve UI for non-API paths (API routes are matched first)
            if full_path.startswith("api") or full_path == "health" or full_path.startswith("docs") or full_path.startswith("openapi"):
                raise HTTPException(status_code=404, detail="Not found")
            path = (ui_path / full_path).resolve() if full_path else ui_path
            if full_path and path.is_file() and path.parent.resolve().is_relative_to(ui_path.resolve()):
                return FileResponse(path)
            index = ui_path / "index.html"
            if index.is_file():
                return FileResponse(index)
            raise HTTPException(status_code=404, detail="Not found")

    return app


# For running with uvicorn
app = create_app()


def main() -> None:
    """Entry point for philosiphai-server console script."""
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
