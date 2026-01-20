"""FastAPI application for PhilosophAI."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .ws import router as ws_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PhilosophAI",
        description="Knowledge Graph RAG API",
        version="0.1.0",
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(ws_router, prefix="/api")

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Query endpoint (non-streaming)
    @app.post("/api/query")
    async def query(request: dict):
        """
        Execute a query against the knowledge graph.
        
        This is a placeholder - wire up the MultiHopAgent here.
        """
        question = request.get("question", "")
        # TODO: Initialize agent and execute query
        return {
            "answer": "Agent not initialized. See src/api/main.py",
            "question": question,
        }

    # Graph data endpoint for UI
    @app.get("/api/graph")
    async def get_graph_data():
        """
        Return graph data for 3D visualization.
        
        This is a placeholder - load from graph builder.
        """
        # TODO: Load graph and return nodes/links for force-graph
        return {
            "nodes": [],
            "links": [],
        }

    # Communities endpoint
    @app.get("/api/communities")
    async def get_communities():
        """Return all communities with summaries."""
        # TODO: Load from storage
        return {"communities": []}

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
