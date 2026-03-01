# PhilosophAI (philosiphai)

**Knowledge Graph RAG with community routing** — a GraphRAG-style system that builds a knowledge graph from philosophical texts, detects communities, and answers questions via an agent that traverses the graph and synthesizes answers with citations.

## Features

- **Ingest pipeline**: Assumes **triples and source data are already in the database** (created by a separate process). The pipeline then: clean triples → canonicalize entities → embed chunks → detect communities (Leiden) → generate community reports
- **Graph storage**: DuckDB-backed storage for triples, chunks, embeddings, and community metadata
- **Agentic Q&A**: Philosopher agent with sequential thinking, phase-gated tools, and multi-hop traversal over the graph
- **REST + WebSocket API**: Query and agent endpoints, graph/communities data, export (transcript, report HTML)
- **Web UI**: Chat interface with graph visualization (static `ui/` served by backend or via separate dev server)

## Requirements

- **Python** 3.11+
- **OpenAI API key** (for embeddings, canonicalization, community summaries, and the philosopher agent)
- **DuckDB database** (path set via `PHILOSOPH_DB`); triples and source data are **created beforehand** by your own pipeline — this repo’s ingest then cleans, canonicalizes, embeds, and builds communities on top of that.

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd philosiphai
pip install .
# Or with dev deps: pip install -e ".[dev]"
```

### 2. Environment

Copy `.env.example` to `.env` and set:

```bash
# Required
OPENAI_API_KEY=sk-...
PHILOSOPH_DB=/path/to/your/philosoph.duckdb

# Optional (defaults shown)
PHILOSOPH_ENV=development
PORT=8000
```

See `.env.example` for logging, rate limits, traversal limits, and CORS.

### 3. Ingest (build the graph)

Run the pipeline against your DuckDB file. **Triples and source data must already exist in the database** (created beforehand by your own extraction/ETL). The CLI then runs:

```bash
# Full pipeline: clean → canonicalize → embed → communities → reports
python -m src.ingest.cli --db /path/to/philosoph.duckdb --all

# Or step by step
python -m src.ingest.cli --db /path/to/philosoph.duckdb --clean
python -m src.ingest.cli --db /path/to/philosoph.duckdb --canonicalize
python -m src.ingest.cli --db /path/to/philosoph.duckdb --embed
python -m src.ingest.cli --db /path/to/philosoph.duckdb --communities
python -m src.ingest.cli --db /path/to/philosoph.duckdb --reports
```

CLI options: `--resolution`, `--min-edge-weight`, `--resolution-sweep`, `--dry-run`, etc. Run `python -m src.ingest.cli --help` for details.

### 4. Run the API server

```bash
# From project root
philosiphai-server
# Or: uvicorn src.api.main:app --reload --port 8000
```

- API: http://localhost:8000  
- Health: http://localhost:8000/health  
- OpenAPI: http://localhost:8000/docs  

The app serves the UI from `/` when running in production mode; in development you can serve the `ui/` folder separately (e.g. with a static server or a Vite app if you add one) and point it at the backend.

## API overview

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness/readiness |
| `POST /api/query` | One-shot RAG query (answer + citations + traversal) |
| `POST /api/agent/query` | Agentic query (multi-step, conversation_id for session) |
| `POST /api/agent/reset` | Reset agent session (per conversation or shared) |
| `GET /api/agent/greeting` | Greeting message for the agent |
| `GET /api/graph` | Graph structure for visualization |
| `GET /api/communities` | List communities |
| `GET /api/communities/{id}` | Single community details |
| `POST /api/export/transcript` | Export transcript as text |
| `POST /api/export/report-html` | Export report as HTML |
| `POST /api/export/field-html` | Export field as HTML |

WebSocket endpoints are mounted under the same app for real-time agent interaction (see `src.api.ws`).

## Project layout

```
philosiphai/
├── src/
│   ├── api/          # FastAPI app, routes, WebSockets, rate limit, MCP server
│   ├── agents/       # Philosopher agent, tools, phases, multi-hopper, trace
│   ├── config/       # Logging, env
│   ├── graph/        # Graph build, communities, traversal, reports, conceptness
│   ├── ingest/       # CLI, cleaner, canonicalizer, embedder
│   ├── rag/          # Vector search, fusion, seeds, citations
│   ├── schema/       # Data models
│   └── storage/      # DuckDB storage
├── ui/               # Frontend (e.g. index.html + assets)
├── tests/
├── pyproject.toml
├── Dockerfile
├── entrypoint.sh     # DB download + uvicorn
├── fly.toml          # Fly.io deployment
├── .env.example
└── DEPLOY.md         # Security checklist, Fly.io commands, CORS
```

## Development

- **Tests**: `pytest` (config in `pyproject.toml`). Markers: `e2e`, `live_integration`, `thread_safety`, etc. For live integration, set `RUN_LIVE_INTEGRATION=1` and have `OPENAI_API_KEY` and `PHILOSOPH_DB` set.
- **Linting/formatting**: `ruff`, `black` (see optional dev deps).
- **Local CORS**: Set `PHILOSOPH_ENV=development` in `.env` so default CORS allows localhost; override with `CORS_ORIGINS` if needed.

## Deployment

- **Docker**: Build from the repo root; image runs `entrypoint.sh` (optional DB download via `DB_DOWNLOAD_URL`) then uvicorn. Set `OPENAI_API_KEY` and `PHILOSOPH_DB` (e.g. `/data/philosoph.duckdb`) via env.
- **Fly.io**: See `DEPLOY.md` for app create, volume, secrets, and CORS. Production env and CORS defaults are in `fly.toml` and `[env]`.

## License

Proprietary (see `pyproject.toml` and `LICENSE`).
