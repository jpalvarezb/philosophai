"""FastAPI application for PhilosophAI with GraphRAG."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ws import router as ws_router, set_agent, set_philosopher_agent


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
        self.ready = False


state = AppState()


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
    db_path = os.environ.get("PHILOSOPH_DB", "data/philosoph.duckdb")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    print(f"📂 Initializing with DB: {db_path}")

    # Storage
    state.storage = DuckDBStorage(Path(db_path))

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
        version="0.6.2",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include WebSocket router
    app.include_router(ws_router, prefix="/api")

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": state.ready}

    # Query endpoint (non-streaming)
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
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

    # Agentic query endpoint (CrewAI-based with sequential thinking)
    @app.post("/api/agent/query", response_model=AgentQueryResponse)
    async def agent_query(request: AgentQueryRequest):
        """
        Execute an agentic query with sequential thinking.

        This endpoint uses CrewAI to orchestrate tool calling with
        explicit reasoning steps. The agent:
        1. Determines appropriate scope by calling list_available_sources
        2. Sets scope if needed, or skips for broad queries
        3. Searches for relevant evidence
        4. Explores the knowledge graph
        5. Synthesizes an answer with citations

        All reasoning is documented via the sequential_thinking tool.
        """
        if not state.ready or not state.philosopher_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        try:
            result = state.philosopher_agent.query(
                question=request.question,
                max_iterations=request.max_iterations,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Agentic greeting (LLM-generated)
    @app.post("/api/agent/reset")
    async def agent_reset():
        if not state.ready or not state.philosopher_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        try:
            state.philosopher_agent.reset_session()
            return {"status": "ok", "message": "Session reset"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/agent/greeting")
    async def agent_greeting():
        if not state.ready or not state.philosopher_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        try:
            return {"greeting": state.philosopher_agent.generate_greeting()}
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
            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "community": state.node_to_community.get(node_id),
                "degree": deg_map.get(node_id, 0),
            })

        for u, v, data in H.edges(data=True):
            if u in node_set and v in node_set:
                links.append({
                    "source": u,
                    "target": v,
                    "label": data.get("label", ""),
                    "weight": data.get("weight", 1),
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

    # Mount static UI files (if ui/ exists)
    ui_path = Path(__file__).parent.parent.parent / "ui"
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="ui")

    return app


# For running with uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
