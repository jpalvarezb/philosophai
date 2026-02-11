"""Phase definitions and tool access control for PhilosophAI agent.

Defines the query processing phases and which tools are allowed in each phase.
Used by PhilosopherAgent to enforce the state machine.
"""
from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """Query processing phases for state machine enforcement."""
    SESSION = "session"
    GUARD = "guard"
    SCOPE = "scope"
    RETRIEVAL = "retrieval"
    TRAVERSAL = "traversal"
    SYNTHESIS = "synthesis"
    DONE = "done"  # Terminal state after synthesis


# Phase-gated tool access (no critique phase — TRAVERSAL → SYNTHESIS only)
PHASE_TOOLS = {
    Phase.SESSION: {
        "sequential_thinking",
        "detect_followup",
        "guard_relevance",
        "skip_scope",
        "skip_guard",
    },
    Phase.GUARD: {
        "sequential_thinking",
        "guard_relevance",
        "skip_scope",
        "skip_guard",
    },
    Phase.SCOPE: {
        "sequential_thinking",
        "list_available_sources",
        "set_scope",
        "clear_scope",
        "skip_scope",
    },
    Phase.RETRIEVAL: {
        "sequential_thinking",
        "search_vectors",
        "search_community_reports",
        "read_community_summary",
        "get_seed_entities_for_communities",
        "get_collected_chunks",
        "get_entities_from_chunks",
        "get_chunk_content",
        "advance_to_traversal",
        "advance_to_synthesis",
    },
    Phase.TRAVERSAL: {
        "sequential_thinking",
        "get_seed_entities_for_communities",
        "get_collected_chunks",
        "expand_node",
        "backtrack",
        "get_traversal_state",
        "get_chunk_content",
        "advance_to_synthesis",
    },
    Phase.SYNTHESIS: {
        "sequential_thinking",
        "synthesize_answer",
    },
    Phase.DONE: set(),
}
