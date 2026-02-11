"""WebSocket endpoints for streaming agent updates."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from ..agents import MultiHopAgent, PhilosopherAgent

router = APIRouter()

# Agent instance (set by main.py at startup)
_agent: "MultiHopAgent | None" = None
_philosopher_agent: "PhilosopherAgent | None" = None


def set_agent(agent: "MultiHopAgent"):
    """Set the agent instance for WebSocket handlers."""
    global _agent
    _agent = agent


def set_philosopher_agent(agent: "PhilosopherAgent"):
    """Set the philosopher agent instance for WebSocket handlers."""
    global _philosopher_agent
    _philosopher_agent = agent


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            await connection.send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query execution.
    
    Client sends: {"question": "...", "max_hops": 2, "use_community_routing": true}
    Server streams: {"type": "status|thought|routing|traversal|answer|complete", ...}
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            request = json.loads(data)
            question = request.get("question", "")
            max_hops = request.get("max_hops", 2)
            use_community_routing = request.get("use_community_routing", True)

            if not question:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "No question provided",
                })
                continue

            if not _agent:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Agent not initialized",
                })
                continue

            # Create event emitter that sends to websocket
            async def emit_event(event: dict):
                await manager.send_json(websocket, event)

            # Run query with streaming events
            # We need to run the sync agent in a thread to not block
            loop = asyncio.get_event_loop()
            
            def run_query():
                events = []
                def on_event(e):
                    events.append(e)
                
                # Temporarily set event handler
                old_handler = _agent.on_event
                _agent.on_event = on_event
                
                try:
                    result = _agent.query(
                        question=question,
                        max_hops=max_hops,
                        use_community_routing=use_community_routing,
                    )
                    return result, events
                finally:
                    _agent.on_event = old_handler
            
            # Run in thread pool
            result, events = await loop.run_in_executor(None, run_query)
            
            # Send collected events
            for event in events:
                await manager.send_json(websocket, event)
            
            # Send routing info
            trace = result.get("trace", {})
            routing = trace.get("routing", {})
            if routing.get("selected_communities"):
                await manager.send_json(websocket, {
                    "type": "routing",
                    "communities": routing["selected_communities"],
                    "scores": routing.get("community_scores", {}),
                })
            
            # Send traversal info for highlighting
            traversal = result.get("traversal", {})
            if traversal:
                await manager.send_json(websocket, {
                    "type": "traversal",
                    "visited_nodes": traversal.get("visited_nodes", [])[:50],
                    "visited_edges": traversal.get("visited_edges", [])[:50],
                    "communities": traversal.get("visited_communities", []),
                })
            
            # Send final result
            await manager.send_json(websocket, {
                "type": "complete",
                "answer": result["answer"],
                "citations": result["citations"],
                "trace": trace,
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        try:
            await manager.send_json(websocket, {
                "type": "error",
                "message": str(e),
            })
        except:
            pass
        manager.disconnect(websocket)


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agentic query execution.
    
    Client sends: {"question": "...", "max_iterations": 25}
    Server streams: {"type": "status|thought|routing|traversal|complete|error", ...}
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            question = request.get("question", "")
            max_iterations = request.get("max_iterations", 25)

            if not question:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "No question provided",
                })
                continue

            if not _philosopher_agent:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Agent not initialized",
                })
                continue

            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[dict] = asyncio.Queue()
            done = asyncio.Event()

            def emit_event(event: dict):
                loop.call_soon_threadsafe(queue.put_nowait, event)

            async def sender():
                while True:
                    if done.is_set() and queue.empty():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    await manager.send_json(websocket, event)

            sender_task = asyncio.create_task(sender())

            def run_query():
                try:
                    _philosopher_agent.query_streaming(
                        question=question,
                        on_event=emit_event,
                        max_iterations=max_iterations,
                    )
                except Exception as e:
                    emit_event({"type": "error", "message": str(e)})
                finally:
                    loop.call_soon_threadsafe(done.set)

            await loop.run_in_executor(None, run_query)
            await sender_task

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        try:
            await manager.send_json(websocket, {
                "type": "error",
                "message": str(e),
            })
        except:
            pass
        manager.disconnect(websocket)


@router.websocket("/ws/graph")
async def websocket_graph(websocket: WebSocket):
    """
    WebSocket for real-time graph highlighting.
    
    Server pushes: {"type": "highlight", "nodes": [...], "edges": [...], "communities": [...]}
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, push updates as agent traverses
            data = await websocket.receive_text()
            # Echo for now
            await manager.send_json(websocket, {"type": "ack", "received": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
