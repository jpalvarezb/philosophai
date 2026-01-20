"""Trace recording for agent execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class ThoughtStep:
    """A single step in the agent's reasoning."""
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class TraceRecorder:
    """Records the agent's thinking process for debugging and UI display."""
    query: str
    thoughts: list[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    citations_used: list[int] = field(default_factory=list)
    total_chunks_retrieved: int = 0
    communities_explored: list[int] = field(default_factory=list)
    nodes_visited: list[str] = field(default_factory=list)
    
    # Phase 3: Enhanced trace fields for GraphRAG routing
    selected_community_ids: list[int] = field(default_factory=list)  # Communities from report search
    community_report_scores: dict[int, float] = field(default_factory=dict)  # comm_id -> score
    seed_entities: list[str] = field(default_factory=list)  # Seed nodes used for traversal
    final_cited_chunk_ids: list[str] = field(default_factory=list)  # Chunks in final answer
    paths: list[dict] = field(default_factory=list)  # Traversal paths: [{node_ids: [], edge_labels: []}]

    def add_thought(
        self,
        thought: str,
        action: str | None = None,
        action_input: dict | None = None,
        observation: str | None = None,
    ):
        """Record a thinking step."""
        step = ThoughtStep(
            step_number=len(self.thoughts) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        self.thoughts.append(step)

    def set_answer(self, answer: str, citations: list[int]):
        """Record the final answer."""
        self.final_answer = answer
        self.citations_used = citations

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "query": self.query,
            "thoughts": [
                {
                    "step": t.step_number,
                    "thought": t.thought,
                    "action": t.action,
                    "action_input": t.action_input,
                    "observation": t.observation[:500] if t.observation else None,
                    "timestamp": t.timestamp,
                }
                for t in self.thoughts
            ],
            "final_answer": self.final_answer,
            "citations_used": self.citations_used,
            "stats": {
                "total_chunks": self.total_chunks_retrieved,
                "communities_explored": self.communities_explored,
                "nodes_visited_count": len(set(self.nodes_visited)),
            },
            # Phase 3: GraphRAG routing trace
            "routing": {
                "selected_communities": self.selected_community_ids,
                "community_scores": self.community_report_scores,
                "seed_entities": self.seed_entities[:20],  # Cap for readability
                "final_cited_chunks": self.final_cited_chunk_ids,
            },
            "paths": self.paths,
        }

    def get_streaming_events(self) -> list[dict]:
        """Get events formatted for WebSocket streaming."""
        events = []
        for thought in self.thoughts:
            events.append({
                "type": "thought",
                "step": thought.step_number,
                "content": thought.thought,
            })
            if thought.action:
                events.append({
                    "type": "action",
                    "step": thought.step_number,
                    "action": thought.action,
                    "input": thought.action_input,
                })
            if thought.observation:
                events.append({
                    "type": "observation",
                    "step": thought.step_number,
                    "content": thought.observation[:200] + "..." if len(thought.observation) > 200 else thought.observation,
                })
        return events
