"""Philosopher agent with sequential thinking and phase-gated tools.

This agent uses OpenAI function calling to orchestrate tool calling with explicit
reasoning steps via the sequential_thinking tool.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from openai import OpenAI

from .phases import Phase, PHASE_TOOLS
from ..config.logging import trace_logger, scope_logger

if TYPE_CHECKING:
    from .tools import AgentTools, ToolResult
    from ..rag import CitationBuilder


# OpenAI tool schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sequential_thinking",
            "description": """Record a step in your sequential thinking process.

Use this tool to structure your reasoning, revise previous thoughts,
and branch into different lines of inquiry. Call this BEFORE and AFTER
other tool calls to document your reasoning.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your current thought or reasoning step"},
                    "thought_number": {"type": "integer", "description": "Current thought number (1-indexed)"},
                    "total_thoughts": {"type": "integer", "description": "Estimated total thoughts needed (can increase)"},
                    "next_thought_needed": {"type": "boolean", "description": "Whether another thought is needed after this"},
                    "is_revision": {"type": "boolean", "description": "Whether this revises a previous thought"},
                    "revises_thought": {"type": "integer", "description": "If is_revision, which thought number is being revised"},
                    "branch_from_thought": {"type": "integer", "description": "If branching, which thought to branch from"},
                    "branch_id": {"type": "string", "description": "Identifier for this branch"},
                    "needs_more_thoughts": {"type": "boolean", "description": "Set to true if you need more thoughts"},
                },
                "required": ["thought", "thought_number", "total_thoughts", "next_thought_needed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_followup",
            "description": "Decide if the new query is a follow-up to recent QA context. Return IN if follow-up (reuse state), OUT if new topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "recent_qas": {"type": "array", "items": {"type": "string"}, "description": "Up to 5 recent Q/A pairs as strings"},
                },
                "required": ["question", "recent_qas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guard_relevance",
            "description": "Decide if the query fits Phil's domain (philosophy/theology/history/culture). Return a short verdict; if out-of-domain, tell the user to ask within scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Original user question"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_guard",
            "description": "Bypass guard when you are certain the query is in-domain.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_community_reports",
            "description": "Search community report summaries to route broad/unscoped queries into relevant communities. Returns community_ids, scores, and cited_chunk_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "limit": {"type": "integer", "description": "Maximum communities to return (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_community_summary",
            "description": "Read the summary, size, and top terms of a community to understand its topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "community_id": {"type": "integer", "description": "Community ID to read"},
                },
                "required": ["community_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_sources",
            "description": """List available sources for scoping. Call this to discover what
authors, titles, traditions, or domains exist in the knowledge base.

IMPORTANT: Call this BEFORE set_scope to ensure you use valid values.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["authors", "titles", "traditions", "domains"],
                        "description": "Category of sources to list",
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scope",
            "description": """Restrict retrieval to specific authors, works, traditions, or domains.

Use when the query asks about a specific philosopher, text, or tradition.
DO NOT use for comparative or open-ended questions.

IMPORTANT: Call list_available_sources first to get valid values.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "authors": {"type": "array", "items": {"type": "string"}, "description": "Author names"},
                    "titles": {"type": "array", "items": {"type": "string"}, "description": "Work titles"},
                    "traditions": {"type": "array", "items": {"type": "string"}, "description": "Traditions"},
                    "domains": {"type": "array", "items": {"type": "string"}, "description": "Domains"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_scope",
            "description": "Skip scoping and proceed to retrieval with global search. Use for comparative or open-ended questions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vectors",
            "description": "Search for relevant chunks using vector similarity. If scope was set, only searches within scoped texts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "limit": {"type": "integer", "description": "Maximum chunks to return (default: 15)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entities_from_chunks",
            "description": "Find entities mentioned in chunks, scored by query relevance. Returns entities sorted by score with breakdown (query token overlap, community match). Use the entity_id values for expand_node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunk IDs to extract entities from"},
                    "query": {"type": "string", "description": "The user's question - used to score entity relevance"},
                },
                "required": ["chunk_ids", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_content",
            "description": "Retrieve the actual text content for chunks. Use to read the evidence supporting your reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunk IDs to retrieve"},
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_node",
            "description": "Expand a node to see neighbors with traversal scores. Updates your current position in the graph. Each neighbor shows: traversal_score (0-1, higher=better path), score_breakdown (edge_wt, evidence, in_scope), predicate (relationship type). Use high-scoring in-scope neighbors for exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Entity ID to expand (use exact IDs from get_entities_from_chunks)"},
                    "max_neighbors": {"type": "integer", "description": "Maximum neighbors to return (default: 10)"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtrack",
            "description": "Return to a previously visited node in your traversal path. Use when a path leads to dead ends or irrelevant concepts. You can backtrack to any node in your history, not just the immediate previous one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "description": "Number of steps to go back (default: 1). Use 0 to see current state without moving."},
                    "to_node_id": {"type": "string", "description": "Alternatively, specify exact node ID to return to (must be in traversal history)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_traversal_state",
            "description": "Get your current traversal state: current node, path taken, and nodes you can backtrack to. Useful for planning next moves.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_to_synthesis",
            "description": "Advance to the synthesis phase to provide your final answer. Call this when you have gathered enough evidence from retrieval and traversal.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_answer",
            "description": "Provide the final answer with citations. ONLY available after calling advance_to_synthesis. This ends the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Your answer with inline citations like [1], [2]"},
                    "cited_chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunk IDs used as evidence, in citation order"},
                },
                "required": ["answer", "cited_chunk_ids"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are **Phil**, a scholarly knowledge agent for philosophy, theology, history, and culture.
You reason rigorously, cite evidence, and traverse a knowledge graph of concepts, authors, works, and ideas.

SAFETY & ETHICS (CRITICAL)
- If a user expresses self-harm, suicide, or violence: respond briefly with empathy, urge contacting emergency services or crisis lines; do NOT debate or encourage.
- For major personal decisions (e.g., divorce, medical/mental health): offer general philosophical/theological perspectives, but avoid directives; suggest consulting qualified professionals.
- Never encourage harmful, illegal, or self-destructive actions. Keep a neutral, academic tone.

WORKFLOW (phase-gated)
0) SESSION
   - If this is a follow-up, continue prior context but go to RETRIEVAL to find new evidence.
   - If new topic, start fresh (GUARD).
   - Call guard_relevance(query) first. If out-of-domain, stop with a brief redirection. If in-domain, proceed to scope.
2) SCOPE (must be first)
   - Specific author/text/tradition -> set_scope
   - Broad/comparative/open-ended -> skip_scope
   - list_available_sources if you need valid scope values
3) RETRIEVAL
   - Unscoped/global: search_community_reports(query) to find relevant communities -> read_community_summary to check topics -> search_vectors
   - Scoped: go straight to search_vectors(query) (communities are derived from scope)
   - get_entities_from_chunks(chunk_ids, query) to score seeds; get_chunk_content to read evidence
4) TRAVERSAL (graph exploration)
   - expand_node on high-scoring seeds
   - If path is weak or node_in_scope is false -> backtrack, use get_traversal_state if lost
   - Continue until you have enough evidence, then advance_to_synthesis
5) SYNTHESIS
   - synthesize_answer with citations [1], [2], ... matching cited_chunk_ids order

SCORES & HOW TO USE THEM
- Community Search (search_community_reports):
  - scores = cosine similarity of query to community summary embedding.
  - Returns IDs only. Call read_community_summary(id) to see what the community is about.
- Seed scoring (get_entities_from_chunks):
  - query_relevance: token overlap with the question
  - in_scope_degree: # of in-scope edges; if 0 (scoped), drop it
  - phrase_penalty: downweights multi-word extraction artifacts
  - final score = relevance – penalty + degree bonus (up to +0.3)
- Traversal scoring (expand_node):
  - traversal_score = edge_weight term + evidence from scoped_chunk_count + in_scope bonus (+0.3)
  - in_scope is true only if (subj,obj,pred) is in scoped_edges (canonical IDs)
  - node_in_scope false => no in-scope edges: backtrack
- GraphTraverser (reference): prioritizes query token overlap, community affinity, edge weight, depth penalty; it is structural (no embeddings in traversal).

SCOPE & INDUCED SUBGRAPH
- scoped_edges = edges supported by scoped chunks; induced nodes = V(scoped_edges).
- Only edges in scoped_edges are “in scope”; node_in_scope false means dead end under current scope.

CITATIONS
- Cite only chunks you actually read via get_chunk_content.
- Use [1], [2], ... in answer text; order matches cited_chunk_ids.
- Weave citations inside sentences; never add a standalone \"Sources\" or \"References\" section.

OUTPUT
- Inline citations only; no bullet lists of sources and no trailing source sections.

TRAVERSAL INTENSITY
- Before synthesis, expand at least two promising edges when evidence exists; explore multiple communities if scores are close.
- Backtrack when a path stalls; do not stop after a single hop unless the question is trivial or clearly irrelevant.

STYLE
- Scholarly, precise, high information density. Present multiple viewpoints when relevant. State uncertainty.
- Be concise but thorough enough to answer rigorously; avoid fluff.
"""


class PhilosopherAgent:
    """
    An agentic query handler using OpenAI function calling.

    Unlike the hardcoded MultiHopAgent pipeline, this agent:
    - Uses LLM-driven tool selection
    - Explicitly reasons via sequential_thinking
    - Respects phase-based tool access control
    """

    def __init__(
        self,
        agent_tools: "AgentTools",
        citation_builder: "CitationBuilder | None" = None,
        llm_client: OpenAI | None = None,
        llm_model: str = "gpt-4o",
        verbose: bool = True,
    ):
        self.agent_tools = agent_tools
        self.citation_builder = citation_builder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.verbose = verbose

        # State managed per-query
        self._current_phase = Phase.SESSION
        self._scope = None
        self._collected_chunks: list[str] = []
        self._collected_entities: list[str] = []
        self._traversed_edges: list[dict] = []  # Track edges explored via expand_node
        self._thoughts: list[dict] = []
        self._final_answer: str = ""
        self._citations: list = []
        self._last_qas: list[tuple[str, str]] = []  # keep last 5 (question, answer)
        self._new_edges_count: int = 0  # Track hops explored in the current turn
        self._read_chunk_ids: set[str] = set()  # All chunks read this session
        self._read_chunk_ids_this_turn: set[str] = set()  # Chunks read in the current turn
        self._session_continued_current_turn: bool = False
        # Traversal state for backtracking
        self._traversal_path: list[str] = []  # Stack of visited node IDs
        self._traversal_history: list[dict] = []  # Full history with context
        self._current_node: str | None = None  # Current position in graph

    def reset_session(self):
        """Full reset of the session, including conversation history."""
        self._reset_state()
        self._last_qas = []
        trace_logger.info("[AGENT] 🔄 Session fully reset (history cleared)")

    def _reset_state(self):
        """Reset state for a new query."""
        self._current_phase = Phase.SESSION
        self._scope = None
        self._collected_chunks = []
        self._collected_entities = []
        self._traversed_edges = []
        self._thoughts = []
        self._final_answer = ""
        self._citations = []
        self._new_edges_count = 0
        self._read_chunk_ids = set()
        self._read_chunk_ids_this_turn = set()
        self._session_continued_current_turn = False
        # Reset traversal state
        self._traversal_path = []
        self._traversal_history = []
        self._current_node = None
        # Clear any active scope from previous query
        self.agent_tools.clear_active_scope()

    def generate_greeting(self) -> str:
        """Short, varied self-intro inviting the user to ask a question."""
        if not self.llm_client:
            return "Phil online—I'll walk the graph, cite inline. What should we explore?"
        system_msg = SYSTEM_PROMPT + (
            "\n\nGREET QUICKLY:\n"
            "- Output exactly one sentence, under 22 words.\n"
            "- Introduce yourself as Phil and mention your areas of focus.\n"
            "- End by inviting the user's question.\n"
            "- No citations, markdown, bullets, or lists."
        )
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": "Give the greeting now."},
                ],
                temperature=0.3,
                max_tokens=60,
            )
            text = resp.choices[0].message.content.strip()
            return text[:200]
        except Exception:
            return "Phil here—graph-walking with inline citations. What topic do you want to probe?"

    def _check_phase(self, tool_name: str) -> str | None:
        """Check if tool is allowed in current phase. Returns error message if not."""
        allowed = PHASE_TOOLS.get(self._current_phase, set())
        # Map tool names to phase tool names
        tool_mapping = {
            "detect_followup": "detect_followup",
            "guard_relevance": "guard_relevance",
            "skip_guard": "skip_guard",
            "sequential_thinking": "sequential_thinking",
            "list_available_sources": "list_available_sources",
            "set_scope": "set_scope",
            "skip_scope": "skip_scope",
            "clear_scope": "clear_scope",
            "search_vectors": "search_vectors",
            "search_community_reports": "search_community_reports",
            "get_entities_from_chunks": "get_entities_from_chunks",
            "get_chunk_content": "get_chunk_content",
            "advance_to_traversal": "advance_to_traversal",
            "expand_node": "expand_node",
            "backtrack": "backtrack",
            "get_traversal_state": "get_traversal_state",
            "advance_to_synthesis": "advance_to_synthesis",
            "synthesize_answer": "synthesize_answer",
        }
        mapped_name = tool_mapping.get(tool_name, tool_name)
        if mapped_name not in allowed:
            return (
                f"Tool '{tool_name}' not available in {self._current_phase.value} phase. "
                f"Allowed tools: {sorted(allowed)}"
            )
        return None

    def _advance_phase(self, to_phase: Phase) -> str:
        """Advance to the next phase."""
        valid_transitions = {
            Phase.SESSION: {Phase.GUARD, Phase.SCOPE},
            Phase.GUARD: {Phase.SCOPE},
            Phase.SCOPE: {Phase.RETRIEVAL},
            Phase.RETRIEVAL: {Phase.TRAVERSAL, Phase.SYNTHESIS},
            Phase.TRAVERSAL: {Phase.SYNTHESIS},
            Phase.SYNTHESIS: {Phase.DONE},
            Phase.DONE: set(),  # Terminal - no transitions allowed
        }

        if to_phase not in valid_transitions.get(self._current_phase, set()):
            trace_logger.info(f"[AGENT] ❌ Invalid phase transition: {self._current_phase.value} → {to_phase.value}")
            return f"Cannot transition from {self._current_phase.value} to {to_phase.value}"

        old_phase = self._current_phase
        self._current_phase = to_phase
        trace_logger.info(f"[AGENT] 📍 Phase: {old_phase.value.upper()} → {to_phase.value.upper()}")
        return f"Advanced to {to_phase.value} phase"

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result string."""
        # Check phase
        error = self._check_phase(name)
        if error:
            return error

        if name == "sequential_thinking":
            result = self.agent_tools.sequential_thinking(**arguments)
            self._thoughts.append(result.data)
            return result.message

        elif name == "detect_followup":
            question = arguments.get("question", "")
            recent = arguments.get("recent_qas", [])
            verdict = self._llm_followup_check(question, recent)
            if verdict == "in":
                # continue existing state; jump to traversal if we have a path, else scope
                self._current_phase = Phase.TRAVERSAL if self._traversal_path else Phase.SCOPE
                return "Follow-up detected. Continuing prior context."
            else:
                self._reset_state()
                return "New topic detected. Context reset; starting fresh."

        elif name == "guard_relevance":
            q = arguments.get("query", "")
            if self._is_irrelevant_query(q):
                self._final_answer = self._llm_guard_response(q)
                self._current_phase = Phase.DONE
                return self._final_answer
            if self.llm_client:
                verdict = self._llm_guard_check(q)
                if verdict == "out":
                    self._final_answer = self._llm_guard_response(q)
                    self._current_phase = Phase.DONE
                    return self._final_answer
            self._advance_phase(Phase.SCOPE)
            return "Query is in-domain. Proceeding to scope."

        elif name == "skip_guard":
            self._advance_phase(Phase.SCOPE)
            return "Guard skipped. Proceeding to scope."

        elif name == "read_community_summary":
            result = self.agent_tools.read_community_summary(
                arguments.get("community_id", -1)
            )
            if not result.success:
                return f"Error: {result.message}"
            data = result.data
            return (
                f"Community {data['community_id']} (size {data['size']}):\n"
                f"Top terms: {data['top_terms']}\n"
                f"Summary: {data['summary']}"
            )

        elif name == "list_available_sources":
            result = self.agent_tools.list_available_sources(arguments.get("category", "authors"))
            if not result.success:
                return f"Error: {result.message}"
            items = result.data["items"]
            category = arguments.get("category", "authors")
            if category == "authors":
                lines = [f"- {item['name']} ({item['chunks']} chunks)" for item in items]
            elif category == "titles":
                lines = [f"- {item['title']} by {item['author']} ({item['chunks']} chunks)" for item in items]
            elif category == "traditions":
                lines = [f"- {item['name']} ({item['chunks']} chunks)" for item in items]
            elif category == "domains":
                lines = [f"- {item['name']} ({item['files']} files)" for item in items]
            else:
                lines = [str(item) for item in items]
            return f"Available {category}:\n" + "\n".join(lines)

        elif name == "set_scope":
            result = self.agent_tools.set_scope(
                authors=arguments.get("authors"),
                titles=arguments.get("titles"),
                traditions=arguments.get("traditions"),
                domains=arguments.get("domains"),
            )
            if result.success:
                from .scope import Scope
                self._scope = Scope(
                    authors=arguments.get("authors") or [],
                    titles=arguments.get("titles") or [],
                    traditions=arguments.get("traditions") or [],
                    domains=arguments.get("domains") or [],
                    strict=True,
                )
                # Apply scope to agent_tools for scoped searches
                self.agent_tools.set_active_scope(self._scope)
                scope_logger.info(f"[SCOPE] Set: {self._scope.describe()}")
                scope_logger.info(f"[SCOPE] Chunks: {result.data.get('chunk_count', '?')} | Texts: {result.data.get('text_count', '?')}")
                self._advance_phase(Phase.RETRIEVAL)
            return result.message + " Advanced to retrieval phase."

        elif name == "skip_scope":
            self._scope = None
            self.agent_tools.clear_active_scope()
            self._advance_phase(Phase.RETRIEVAL)
            return "Scope skipped. Advanced to retrieval phase with global search."

        elif name == "search_vectors":
            result = self.agent_tools.search_vectors(
                arguments.get("query", ""),
                limit=arguments.get("limit", 15),
            )
            if not result.success:
                return f"Error: {result.message}"
            chunk_ids = result.data["chunk_ids"]
            scores = result.data["scores"]
            scoped = result.data.get("scoped", False)
            self._collected_chunks.extend(chunk_ids)
            self._collected_chunks = list(dict.fromkeys(self._collected_chunks))
            trace_logger.info(f"[AGENT] 🔍 Vector search: {len(chunk_ids)} chunks (scoped={scoped})")
            lines = [f"- {cid} (score: {score:.3f})" for cid, score in zip(chunk_ids, scores)]
            return f"Found {len(chunk_ids)} chunks:\n" + "\n".join(lines)

        elif name == "search_community_reports":
            result = self.agent_tools.search_community_reports(
                arguments.get("query", ""),
                limit=arguments.get("limit", 5),
            )
            if not result.success:
                return f"Error: {result.message}"
            comm_ids = result.data["community_ids"]
            scores = result.data["scores"]
            cited = result.data.get("cited_chunk_ids", [])
            scoped = result.data.get("scoped", False)
            # Add cited chunks to collected set (they are already scope-filtered if scoped)
            self._collected_chunks.extend(cited)
            self._collected_chunks = list(dict.fromkeys(self._collected_chunks))
            trace_logger.info(f"[AGENT] 🛰️ Community routing: {len(comm_ids)} communities (scoped={scoped})")
            lines = [f"Communities (top {len(comm_ids)}):"]
            for cid, s in zip(comm_ids, scores):
                lines.append(f"  - community {cid} | score={s:.3f}")
            if cited:
                lines.append(f"Cited chunks from reports: {len(cited)}")
            return "\\n".join(lines)

        elif name == "get_entities_from_chunks":
            result = self.agent_tools.get_entities_from_chunks(
                arguments.get("chunk_ids", []),
                query=arguments.get("query", ""),
            )
            if not result.success:
                return f"Error: {result.message}"

            # Handle scored vs unscored response
            if "entities" in result.data:
                # Scored response
                entities = result.data["entities"]
                entity_ids = [e["entity_id"] for e in entities]
                self._collected_entities.extend(entity_ids)
                self._collected_entities = list(dict.fromkeys(self._collected_entities))

                trace_logger.info(f"[AGENT] 🏷️ Extracted {len(entities)} scored entities")
                if entities:
                    top3 = [(e["label"], e["query_relevance"], e["reason"]) for e in entities[:3]]
                    trace_logger.info(f"[AGENT]    Top 3: {top3}")

                # Auto-advance to traversal if we have entities
                if self._current_phase == Phase.RETRIEVAL and entities:
                    self._advance_phase(Phase.TRAVERSAL)

                # Format output with scores
                lines = [f"Found {len(entities)} entities (scored by query relevance):"]
                lines.append(f"Query tokens: {result.data.get('query_tokens', [])}")
                lines.append("")
                for e in entities[:20]:
                    lines.append(
                        f"  - {e['entity_id']} | relevance={e['query_relevance']} | {e['reason']} | community={e['community_id']}"
                    )
                return "\n".join(lines)
            else:
                # Unscored response (no query provided)
                entity_ids = result.data["entity_ids"]
                self._collected_entities.extend(entity_ids)
                self._collected_entities = list(dict.fromkeys(self._collected_entities))
                trace_logger.info(f"[AGENT] 🏷️ Extracted {len(entity_ids)} entities (unscored)")
                if self._current_phase == Phase.RETRIEVAL and entity_ids:
                    self._advance_phase(Phase.TRAVERSAL)
                return f"Found {len(entity_ids)} entities (provide query parameter for relevance scoring)."

        elif name == "get_chunk_content":
            result = self.agent_tools.get_chunk_content(arguments.get("chunk_ids", []))
            if not result.success:
                return f"Error: {result.message}"
            chunks = result.data["chunks"]
            # Track which chunks have been read
            for c in chunks:
                cid = c.get("id")
                if cid:
                    self._read_chunk_ids.add(cid)
                    self._read_chunk_ids_this_turn.add(cid)
            lines = [f"[{c['id']}]\n{c['content']}\n" for c in chunks]
            return "\n---\n".join(lines)

        elif name == "expand_node":
            result = self.agent_tools.expand_node(
                arguments.get("node_id", ""),
                max_neighbors=arguments.get("max_neighbors", 10),
            )
            if not result.success:
                return f"Error: {result.message}"
            data = result.data
            source_id = data["node_id"]
            source_label = data["label"]

            # Update traversal state - track where we are in the graph
            self._current_node = source_id
            if source_id not in self._traversal_path:
                self._traversal_path.append(source_id)
                self._traversal_history.append({
                    "node_id": source_id,
                    "label": source_label,
                    "community_id": data.get("community_id"),
                    "in_scope_neighbors": data.get("in_scope_neighbors", 0),
                    "step": len(self._traversal_path),
                })

            # Track explored edges (only in-scope ones for strict scope)
            added_in_scope = 0
            for n in data["neighbors"]:
                if n.get("in_scope", True):  # Only track in-scope edges
                    self._traversed_edges.append({
                        "source": source_id,
                        "target": n["node_id"],
                        "label": n["predicate"],
                    })
                    added_in_scope += 1

            # Count one hop per successful expansion (not per neighbor listed)
            if added_in_scope > 0:
                self._new_edges_count += 1

            # Build output with scope info
            node_in_scope = data.get("node_in_scope", True)
            trace_logger.info(f"[AGENT] 🕸️ Expand node: {data['label']} (community={data['community_id']}, in_scope={node_in_scope})")
            trace_logger.info(f"[AGENT]    Edges: {data.get('total_neighbors', len(data['neighbors']))} total, {data.get('in_scope_neighbors', '?')} in-scope")
            trace_logger.info(f"[AGENT]    Path: {' → '.join(self._traversal_path[-5:])}")

            lines = [f"Node: {data['label']} (community: {data['community_id']})"]
            lines.append(f"Traversal position: step {len(self._traversal_path)} | path: {' → '.join(self._traversal_path[-3:])}")

            # Warn if node is not in induced subgraph
            if data.get("scope_active") and not node_in_scope:
                lines.append("⚠️ WARNING: This node is NOT in the scoped induced subgraph - it has no in-scope edges.")
                lines.append("   Consider using backtrack to return to a previous node.")

            if data.get("scope_active"):
                lines.append(f"Total edges: {data['total_neighbors']} ({data['in_scope_neighbors']} in-scope, {data['out_scope_neighbors']} out-of-scope)")

            lines.append(f"Neighbors ({len(data['neighbors'])} shown):")
            for n in data["neighbors"]:
                scope_marker = "" if n.get("in_scope", True) else " [OUT OF SCOPE]"
                chunk_info = f"evidence: {n.get('scoped_chunk_count', n['chunk_count'])}"
                score_info = f"score={n.get('traversal_score', '?')}"
                breakdown = n.get('score_breakdown', '')
                lines.append(
                    f"  - {n['node_id']} | {score_info} ({breakdown}) | via '{n['predicate']}' | {chunk_info}{scope_marker}"
                )
            return "\n".join(lines)

        elif name == "backtrack":
            steps = arguments.get("steps", 1)
            to_node_id = arguments.get("to_node_id")

            if not self._traversal_path:
                return "Error: No traversal history to backtrack through. Use expand_node first."

            # If specific node requested, find it
            if to_node_id:
                if to_node_id not in self._traversal_path:
                    return f"Error: Node '{to_node_id}' not in traversal history. History: {self._traversal_path}"
                # Truncate path to that node
                idx = self._traversal_path.index(to_node_id)
                self._traversal_path = self._traversal_path[:idx + 1]
                self._current_node = to_node_id
            elif steps == 0:
                # Just show state without moving
                pass
            else:
                # Go back N steps
                if steps >= len(self._traversal_path):
                    # Go back to start
                    self._traversal_path = self._traversal_path[:1]
                else:
                    self._traversal_path = self._traversal_path[:-steps]
                self._current_node = self._traversal_path[-1] if self._traversal_path else None

            # Build response
            if self._current_node:
                # Find the history entry for current node
                current_info = next(
                    (h for h in self._traversal_history if h["node_id"] == self._current_node),
                    {"label": self._current_node, "community_id": "?"}
                )
                trace_logger.info(f"[AGENT] ↩️ Backtrack to: {current_info.get('label', self._current_node)}")
                lines = [
                    f"Backtracked to: {current_info.get('label', self._current_node)} (community: {current_info.get('community_id', '?')})",
                    f"Current path: {' → '.join(self._traversal_path)}",
                    f"You can now expand_node on '{self._current_node}' to see its neighbors again, or explore a different path.",
                ]
            else:
                lines = ["Backtracked to start. No current node. Use expand_node to begin traversal."]

            return "\n".join(lines)

        elif name == "get_traversal_state":
            lines = ["=== Traversal State ==="]

            if not self._traversal_path:
                lines.append("No traversal started. Use expand_node to begin.")
            else:
                lines.append(f"Current node: {self._current_node}")
                lines.append(f"Path length: {len(self._traversal_path)} nodes")
                lines.append(f"Path: {' → '.join(self._traversal_path)}")
                lines.append("")
                lines.append("History (can backtrack to any):")
                for h in self._traversal_history:
                    marker = "▶" if h["node_id"] == self._current_node else " "
                    lines.append(
                        f"  {marker} [{h['step']}] {h['label']} (community={h['community_id']}, in_scope_edges={h['in_scope_neighbors']})"
                    )

            lines.append("")
            lines.append(f"Edges explored: {len(self._traversed_edges)}")
            lines.append(f"Entities collected: {len(self._collected_entities)}")

            return "\n".join(lines)

        elif name == "advance_to_synthesis":
            if self._current_phase in (Phase.RETRIEVAL, Phase.TRAVERSAL):
                # Ensure we have actively traversed new edges this turn
                if self._new_edges_count < 2 and self._collected_entities:
                    return "Traversal too shallow—expand_node along at least two NEW promising edges before synthesis."
                # Ensure we have read evidence this turn (prevents generic, ungrounded repeats)
                if not self._read_chunk_ids_this_turn:
                    return "No new evidence read yet—call get_chunk_content on the most relevant chunks before synthesis."
                self._advance_phase(Phase.SYNTHESIS)
                return "Advanced to synthesis phase. Now call synthesize_answer with your response."
            else:
                return f"Cannot advance to synthesis from {self._current_phase.value} phase."

        elif name == "synthesize_answer":
            answer = arguments.get("answer", "")
            cited_chunk_ids = arguments.get("cited_chunk_ids", [])

            # Must cite something (inline citations)
            if not cited_chunk_ids:
                self._current_phase = Phase.TRAVERSAL
                return "Error: Provide cited_chunk_ids for inline citations. Return to traversal, read evidence, then try again."

            # Validate citations: must reference chunks that were actually read
            unread = set(cited_chunk_ids) - set(self._read_chunk_ids)
            if unread:
                self._current_phase = Phase.TRAVERSAL
                return f"Error: These chunk IDs were not read via get_chunk_content: {list(unread)[:5]}"

            # Follow-up strictness: require at least one newly-read chunk this turn
            if self._session_continued_current_turn:
                if not (set(cited_chunk_ids) & self._read_chunk_ids_this_turn):
                    self._current_phase = Phase.TRAVERSAL
                    return "Error: Follow-up answers must cite at least one NEW chunk read this turn. Read new evidence, then try again."

            self._final_answer = answer

            # Build proper citations
            if self.citation_builder and cited_chunk_ids:
                self._citations = self.citation_builder.build_citations(cited_chunk_ids)
            else:
                self._citations = cited_chunk_ids

            # Transition to DONE (terminal state) - not SYNTHESIS (we're already there)
            self._current_phase = Phase.DONE
            trace_logger.info(f"[AGENT] 📍 Phase: SYNTHESIS → DONE")
            return f"Answer recorded with {len(cited_chunk_ids)} citations. Query complete."

        return f"Unknown tool: {name}"

    def query(self, question: str, max_iterations: int = 25) -> dict:
        """
        Execute a query using the agentic reasoning loop.

        Args:
            question: The user's question
            max_iterations: Maximum tool calls before stopping

        Returns:
            Dict with answer, citations, and reasoning trace
        """
        normalized_question = question.strip()

        # Track per-turn activity (do not reset full graph state on follow-ups)
        self._new_edges_count = 0
        self._read_chunk_ids_this_turn = set()
        self._session_continued_current_turn = False

        # Reset per-turn outputs so follow-ups cannot reuse the prior answer/citations/thoughts
        self._final_answer = ""
        self._citations = []
        self._thoughts = []

        # Decide whether to reuse prior state (follow-up) or start fresh
        recent_qas_str = [f"Q: {q}\nA: {a}" for q, a in self._last_qas[-5:]]
        followup = self._llm_followup_check(normalized_question, recent_qas_str)
        session_continued = False
        if followup == "out":
            self._reset_state()
            self._current_phase = Phase.GUARD
        else:
            session_continued = True
            self._session_continued_current_turn = True
            # Continue from prior phase/state; always go to RETRIEVAL for follow-ups to get fresh evidence
            # while preserving graph state (_traversal_path, etc.)
            self._current_phase = Phase.RETRIEVAL

        if not self.llm_client:
            raise ValueError("LLM client not provided")

        trace_logger.info(f"[AGENT] ═══════════════════════════════════════════════════════")
        trace_logger.info(f"[AGENT] Query: {question[:100]}{'...' if len(question) > 100 else ''}")
        trace_logger.info(f"[AGENT] Max iterations: {max_iterations}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if followup == "in" and recent_qas_str:
            joined_context = "\n\n".join(recent_qas_str)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Recent conversation context (reuse state):\n{joined_context}\n\n"
                        f"STATUS: You are continuing a session in phase {self._current_phase.value.upper()}. "
                        "Do NOT call guard_relevance or skip_guard. "
                        "Do NOT repeat the prior answer; directly answer the new question with new details and evidence. "
                        "Use retrieval tools (search_vectors) and read evidence (get_chunk_content)."
                    ),
                }
            )
        messages.append({"role": "user", "content": f"Answer this question: {question}"})

        iteration = 0
        done = False

        while not done and iteration < max_iterations:
            iteration += 1

            trace_logger.info(f"[AGENT] ─── Iteration {iteration} | Phase: {self._current_phase.value.upper()} ───")

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="required",
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    # Log tool call
                    if func_name == "sequential_thinking":
                        thought = func_args.get('thought', '')[:150]
                        trace_logger.info(f"[AGENT] 💭 Thought #{func_args.get('thought_number', '?')}: {thought}{'...' if len(func_args.get('thought', '')) > 150 else ''}")
                    else:
                        args_str = json.dumps(func_args, default=str)[:100]
                        trace_logger.tool_call(func_name, args=args_str)

                    result = self._execute_tool(func_name, func_args)

                    # Log result summary
                    result_preview = result[:200] if len(result) > 200 else result
                    if func_name != "sequential_thinking":
                        trace_logger.tool_result(func_name, result=result_preview.replace('\n', ' ')[:150])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                    # Check if we're done (either by name or phase)
                    if self._current_phase == Phase.DONE:
                        trace_logger.info(f"[AGENT] ✅ Synthesis complete")
                        done = True
                        break
            else:
                # No tool calls - model wants to respond directly
                trace_logger.info(f"[AGENT] ⚠️ No tool call from model")

                # If we have an answer, we're done
                if self._final_answer:
                    done = True
                else:
                    # Prompt the model to continue
                    messages.append({
                        "role": "user",
                        "content": "Please continue by calling the appropriate tool. Remember to use sequential_thinking to document your reasoning.",
                    })

        # Build traversal node metadata for nodes seen during traversal/collection
        traversal_node_ids = set(self._collected_entities)
        for e in self._traversed_edges:
            traversal_node_ids.add(e.get("source"))
            traversal_node_ids.add(e.get("target"))
        traversal_nodes_meta = []
        if traversal_node_ids:
            G = self.agent_tools.graph
            community_map = self.agent_tools.node_to_community
            for nid in traversal_node_ids:
                if nid is None:
                    continue
                node_data = G.nodes.get(nid, {}) if G is not None and nid in G else {}
                traversal_nodes_meta.append({
                    "id": nid,
                    "label": node_data.get("label", nid),
                    "community": community_map.get(nid),
                    "degree": G.degree(nid) if G is not None and nid in G else 0,
                })

        # Build response
        citations_data = []
        if self._citations:
            if hasattr(self._citations[0], 'to_dict'):
                citations_data = [c.to_dict() for c in self._citations]
            else:
                citations_data = [{"chunk_id": cid} for cid in self._citations]

        # Persist last QA (keep last 5)
        self._last_qas.append((question, self._final_answer or ""))
        self._last_qas = self._last_qas[-5:]

        return {
            "answer": self._final_answer or "No answer was synthesized within the iteration limit.",
            "citations": citations_data,
            "phase": self._current_phase.value,
            "scope": {
                "authors": self._scope.authors if self._scope else [],
                "titles": self._scope.titles if self._scope else [],
                "traditions": self._scope.traditions if self._scope else [],
                "domains": self._scope.domains if self._scope else [],
            } if self._scope else None,
            "collected_chunks": self._collected_chunks,
            "collected_entities": self._collected_entities,
            "traversal": {
                "visited_nodes": self._collected_entities,
                "edges": self._traversed_edges,
                "edges_traversed": len(self._traversed_edges),
            },
            "traversal_nodes": traversal_nodes_meta,
            "thoughts": self._thoughts,
            "iterations": iteration,
            "session_continued": session_continued,
        }

    def query_streaming(
        self,
        question: str,
        on_event: Callable[[dict], None],
        max_iterations: int = 25,
    ) -> dict:
        """
        Execute a query with streaming events.

        Args:
            question: The user's question
            on_event: Callback for streaming events
            max_iterations: Maximum tool calls before stopping

        Returns:
            Dict with answer, citations, and reasoning trace
        """
        on_event({"type": "status", "message": "Starting agentic query..."})

        # For now, run non-streaming and emit events
        # TODO: Implement true streaming with OpenAI streaming API
        result = self.query(question, max_iterations)

        on_event({"type": "complete", "message": "Query complete"})
        return result

    def _is_irrelevant_query(self, question: str) -> bool:
        """Heuristic guardrail for non-philosophy small talk or off-domain asks."""
        q = question.lower().strip()
        greetings = {"hi", "hello", "hey", "yo", "sup", "hola", "what's up", "whats up"}
        if q in greetings:
            return True
        trivial_prefixes = ("hey ", "hi ", "hello ", "yo ", "sup ")
        if any(q.startswith(p) and len(q.split()) <= 3 for p in trivial_prefixes):
            return True
        off_domain_phrases = [
            "buy me a car", "order me a pizza", "book me a flight", "rent a car",
            "buy me", "price of a car", "i want a car", "want a car", "need a car",
            "buy a car", "purchase a car", "rent a car", "lease a car",
            "book a flight", "book flight", "plane ticket", "plane tickets",
            "order pizza", "buy pizza"
        ]
        if any(p in q for p in off_domain_phrases):
            return True
        # Short commercial-intent queries with vehicle/booking terms
        car_tokens = {"car", "cars", "auto", "vehicle"}
        intent_tokens = {"buy", "purchase", "rent", "lease", "need", "want", "order", "book", "price"}
        words = q.split()
        if len(words) <= 8 and car_tokens & set(words) and intent_tokens & set(words):
            return True
        return False

    def _reject_irrelevant(self, question: str) -> dict:
        """Return a short guardrail response without invoking the LLM."""
        message = "Phil focuses on philosophy, theology, history, and culture. Ask about ideas, thinkers, or texts."
        return {
            "answer": message,
            "citations": [],
            "phase": Phase.DONE.value,
            "scope": None,
            "collected_chunks": [],
            "collected_entities": [],
            "traversal": {"visited_nodes": [], "edges": [], "edges_traversed": 0},
            "traversal_nodes": [],
            "thoughts": [],
            "iterations": 0,
        }

    def _llm_guard_check(self, question: str) -> str:
        """
        Ask the LLM to classify if the query is in Phil's domain.
        Returns "in" or "out".
        """
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify if the user query is within Phil's scope: philosophy, theology, history, culture.\n"
                            "Respond with exactly one token: IN or OUT."
                        ),
                    },
                    {"role": "user", "content": question[:500]},
                ],
                max_tokens=3,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            return "in" if "in" in text[:5] else "out"
        except Exception:
            return "in"

    def _llm_guard_response(self, question: str) -> str:
        """
        LLM-crafted redirect when the query is out of scope.
        Uses SYSTEM_PROMPT for persona consistency; keeps it to <=2 sentences.
        """
        fallback = "Phil focuses on philosophy, theology, history, and culture. Ask about ideas, thinkers, or texts."
        if not self.llm_client:
            return fallback
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "\n\nOUT-OF-SCOPE HANDOFF:\n"
                        "Politely decline and redirect to topics in scope. One or two short sentences. No citations, no markdown."
                    },
                    {"role": "user", "content": question[:500]},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            return (resp.choices[0].message.content or fallback).strip()
        except Exception:
            return fallback

    def _llm_followup_check(self, question: str, recent_qas: list[str]) -> str:
        """
        LLM check to decide if current question is a follow-up.
        Returns "in" (follow-up) or "out" (new topic).
        """
        if not self.llm_client or not recent_qas:
            return "out"
        try:
            context = "\n\n".join(recent_qas[-5:])
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Decide if the new user question is a follow-up to the prior Q&A context. "
                            "Reply with exactly one token: IN (follow-up) or OUT (new topic). "
                            "Be strict; only IN if the new question depends on or continues the prior discussion."
                        ),
                    },
                    {"role": "user", "content": f"Previous QAs:\n{context}\n\nNew question:\n{question}"},
                ],
                max_tokens=3,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            return "in" if "in" in text[:5] else "out"
        except Exception:
            return "out"
