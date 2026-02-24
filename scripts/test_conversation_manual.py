#!/usr/bin/env python3
"""Manual test script for Option B (conversation_id) against a running server.

Usage:
  1. Start the server: philosiphai-server   (or: uvicorn src.api.main:app --reload)
  2. Run: python scripts/test_conversation_manual.py

Requires: httpx (pip install httpx or use dev deps)
"""
from __future__ import annotations

import os
import sys

BASE = os.environ.get("PHILOSOPH_API", "http://localhost:8000")


def main():
    try:
        import httpx
    except ImportError:
        print("Install httpx: pip install httpx")
        sys.exit(1)

    cid = "manual-test-conv-1"

    print(f"Testing conversation_id against {BASE}")
    print()

    with httpx.Client(timeout=60.0) as client:
        # 1) Query without conversation_id (stateless)
        print("1. POST /api/agent/query (no conversation_id)...")
        r = client.post(
            f"{BASE}/api/agent/query",
            json={"question": "What is one sentence on virtue?", "max_iterations": 3},
        )
        if r.status_code != 200:
            print(f"   FAIL: {r.status_code} {r.text}")
            return 1
        data = r.json()
        print(f"   OK answer snippet: {data.get('answer', '')[:120]}...")
        print(f"   session_continued: {data.get('session_continued')}")
        print()

        # 2) Query with conversation_id (first turn)
        print("2. POST /api/agent/query (conversation_id first time)...")
        r = client.post(
            f"{BASE}/api/agent/query",
            json={
                "question": "What is one sentence on virtue?",
                "max_iterations": 3,
                "conversation_id": cid,
            },
        )
        if r.status_code != 200:
            print(f"   FAIL: {r.status_code} {r.text}")
            return 1
        data = r.json()
        print(f"   OK answer snippet: {data.get('answer', '')[:120]}...")
        print(f"   session_continued: {data.get('session_continued')}")
        print()

        # 3) Follow-up with same conversation_id (should reuse agent / last 5 Q/As)
        print("3. POST /api/agent/query (same conversation_id, follow-up)...")
        r = client.post(
            f"{BASE}/api/agent/query",
            json={
                "question": "And in one sentence, what about justice?",
                "max_iterations": 3,
                "conversation_id": cid,
            },
        )
        if r.status_code != 200:
            print(f"   FAIL: {r.status_code} {r.text}")
            return 1
        data = r.json()
        print(f"   OK answer snippet: {data.get('answer', '')[:120]}...")
        print(f"   session_continued: {data.get('session_continued')} (expect True if LLM treated as follow-up)")
        print()

        # 4) Reset that conversation
        print("4. POST /api/agent/reset (conversation_id)...")
        r = client.post(
            f"{BASE}/api/agent/reset",
            json={"conversation_id": cid},
        )
        if r.status_code != 200:
            print(f"   FAIL: {r.status_code} {r.text}")
            return 1
        data = r.json()
        print(f"   OK: {data.get('message')} conversation_id={data.get('conversation_id')}")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
