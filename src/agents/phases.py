"""Phase definitions and tool access control for PhilosophAI agent.

Defines the query processing phases and which tools are allowed in each phase.
Used by PhilosopherAgent to enforce the state machine.
"""
from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """Query processing phases for state machine enforcement."""
    SCOPE = "scope"
    RETRIEVAL = "retrieval"
    TRAVERSAL = "traversal"
    SYNTHESIS = "synthesis"
    DONE = "done"  # Terminal state after synthesis


# Phase-gated tool access
PHASE_TOOLS = {
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
        "get_entities_from_chunks",
        "get_chunk_content",
        "advance_to_traversal",
        "advance_to_synthesis",  # Can skip traversal and go directly to synthesis
    },
    Phase.TRAVERSAL: {
        "sequential_thinking",
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
    Phase.DONE: set(),  # No tools allowed - query is complete
}
