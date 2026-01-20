"""WebSocket endpoints for streaming agent updates."""
from __future__ import annotations

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
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
    
    Client sends: {"question": "..."}
    Server streams: {"type": "status|thought|answer|complete", ...}
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            request = json.loads(data)
            question = request.get("question", "")

            if not question:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "No question provided",
                })
                continue

            # Emit events as agent processes
            # TODO: Wire up MultiHopAgent with on_event callback
            
            await manager.send_json(websocket, {
                "type": "status",
                "message": "Agent not initialized. Wire up MultiHopAgent.",
            })

            await manager.send_json(websocket, {
                "type": "complete",
                "answer": "Placeholder answer",
                "citations": [],
                "traversal": {},
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {
            "type": "error",
            "message": str(e),
        })
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
