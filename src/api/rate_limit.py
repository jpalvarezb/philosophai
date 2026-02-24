"""In-memory rate limiting per client IP (fixed window)."""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.websockets import WebSocket

# Config: set PHILOSOPH_RATE_LIMIT_REQUESTS=0 to disable
_RATE_LIMIT_REQUESTS = int(os.environ.get("PHILOSOPH_RATE_LIMIT_REQUESTS", "2"))
_RATE_LIMIT_WINDOW_SEC = int(os.environ.get("PHILOSOPH_RATE_LIMIT_WINDOW_SECONDS", "60"))

_store: dict[str, tuple[int, float]] = {}  # client_key -> (count, window_start_ts)
_lock = threading.Lock()


def get_client_key(request: "Request") -> str:
    """Client key for rate limiting: X-Forwarded-For (first proxy) or client host."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def get_client_key_ws(websocket: "WebSocket") -> str:
    """Client key for WebSocket: host from connection."""
    if websocket.client:
        return websocket.client.host or "unknown"
    return "unknown"


def record_request(client_key: str) -> bool:
    """
    Record one request for client_key. Returns True if under limit, False if over limit.
    When disabled (RATE_LIMIT_REQUESTS <= 0), always returns True.
    """
    if _RATE_LIMIT_REQUESTS <= 0:
        return True

    now = time.monotonic()
    with _lock:
        # Evict expired windows to avoid unbounded growth
        to_del = [
            k for k, (_, start) in _store.items()
            if now - start >= _RATE_LIMIT_WINDOW_SEC
        ]
        for k in to_del:
            del _store[k]

        if client_key not in _store:
            _store[client_key] = (1, now)
            return True

        count, start = _store[client_key]
        if now - start >= _RATE_LIMIT_WINDOW_SEC:
            _store[client_key] = (1, now)
            return True

        count += 1
        _store[client_key] = (count, start)
        return count <= _RATE_LIMIT_REQUESTS


def get_retry_after_seconds(client_key: str) -> int:
    """Seconds until current window ends (for Retry-After header)."""
    if _RATE_LIMIT_REQUESTS <= 0:
        return 0
    with _lock:
        if client_key not in _store:
            return 0
        _, start = _store[client_key]
        elapsed = time.monotonic() - start
        remaining = max(0, int(_RATE_LIMIT_WINDOW_SEC - elapsed))
        return min(remaining, 60)  # Cap at 60 for header
