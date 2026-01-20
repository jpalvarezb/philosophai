"""Data models for PhilosophAI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text chunk from the corpus."""
    chunk_id: str
    content: str
    text_id: int | None = None
    position: int | None = None
    source_path: str | None = None


@dataclass
class Triple:
    """A subject-predicate-object triple."""
    subject_id: str
    predicate_id: str
    object_id: str
    subject_label: str
    predicate_label: str
    object_label: str
    weight: int = 1
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Community:
    """A cluster of related nodes detected by Leiden."""
    community_id: int
    level: int
    node_ids: list[str]
    size: int
    summary: str | None = None
    summary_embedding: list[float] | None = None
    top_terms: list[str] = field(default_factory=list)


@dataclass
class TraversalStep:
    """A single step in the graph traversal."""
    step_number: int
    node_id: str
    node_label: str
    edge_label: str | None  # Predicate used to reach this node
    from_node_id: str | None  # Previous node
    community_id: int | None
    depth: int = 0  # Hop depth from seed node
    chunk_ids: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class TraversalTrace:
    """Complete trace of a multi-hop traversal for UI highlighting."""
    query: str
    seed_chunks: list[str] = field(default_factory=list)
    seed_nodes: list[str] = field(default_factory=list)
    seed_communities: list[int] = field(default_factory=list)
    steps: list[TraversalStep] = field(default_factory=list)
    visited_nodes: set[str] = field(default_factory=set)
    visited_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (from, to, predicate)
    visited_communities: set[int] = field(default_factory=set)
    collected_chunk_ids: list[str] = field(default_factory=list)
    max_depth: int = 0  # Track max hop depth reached

    def add_step(self, step: TraversalStep):
        self.steps.append(step)
        self.visited_nodes.add(step.node_id)
        if step.community_id is not None:
            self.visited_communities.add(step.community_id)
        if step.from_node_id and step.edge_label:
            self.visited_edges.append((step.from_node_id, step.node_id, step.edge_label))
        self.collected_chunk_ids.extend(step.chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/WebSocket response."""
        # Calculate max depth from steps (use depth field if available, else calculate)
        max_depth = 0
        if self.steps:
            for step in self.steps:
                max_depth = max(max_depth, step.depth)
        
        return {
            "query": self.query,
            "seed_chunks": self.seed_chunks,
            "seed_nodes": self.seed_nodes,
            "seed_communities": list(self.seed_communities),
            "steps": [
                {
                    "step": s.step_number,
                    "node_id": s.node_id,
                    "node_label": s.node_label,
                    "edge_label": s.edge_label,
                    "from_node_id": s.from_node_id,
                    "community_id": s.community_id,
                    "score": s.score,
                }
                for s in self.steps
            ],
            "visited_nodes": list(self.visited_nodes),
            "visited_edges": [
                {"from": e[0], "to": e[1], "predicate": e[2]} for e in self.visited_edges
            ],
            "edges_traversed": len(self.visited_edges),
            "hops": max_depth,
            "visited_communities": list(self.visited_communities),
            "chunk_count": len(set(self.collected_chunk_ids)),
        }


@dataclass
class Citation:
    """An inline citation linking answer text to source chunk."""
    index: int  # [1], [2], etc.
    chunk_id: str
    chunk_content: str
    community_id: int | None = None
    node_ids: list[str] = field(default_factory=list)  # Entities in this chunk

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "preview": self.chunk_content[:300] + "..." if len(self.chunk_content) > 300 else self.chunk_content,
            "full_content": self.chunk_content,
            "community_id": self.community_id,
            "node_ids": self.node_ids,
        }
