"""Fuse and rerank results from vector search and graph traversal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
    from ..schema import TraversalTrace
    from .vector import VectorSearchResult


@dataclass
class FusedResult:
    """Combined results from vector search + graph traversal."""
    chunk_ids: list[str]  # Ordered by relevance
    chunk_scores: dict[str, float]  # chunk_id -> score
    chunk_sources: dict[str, str]  # chunk_id -> "vector" | "graph" | "both"
    trace: "TraversalTrace | None"


class ResultFusion:
    """Merge and rerank results from vector search and graph traversal."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.storage = storage
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def fuse(
        self,
        vector_result: "VectorSearchResult",
        trace: "TraversalTrace | None",
        max_chunks: int = 30,
    ) -> FusedResult:
        """
        Combine vector search results with graph traversal results.
        
        Args:
            vector_result: Results from VectorSearch
            trace: TraversalTrace from graph traversal (or None)
            max_chunks: Maximum chunks to keep
        
        Returns:
            FusedResult with deduplicated, scored chunks
        """
        chunk_scores: dict[str, float] = {}
        chunk_sources: dict[str, str] = {}

        # Add vector search results
        for chunk_id, score in zip(vector_result.chunk_ids, vector_result.chunk_scores):
            chunk_scores[chunk_id] = score * self.vector_weight
            chunk_sources[chunk_id] = "vector"

        # Add graph traversal results
        if trace:
            graph_chunks = list(set(trace.collected_chunk_ids))
            # Score graph chunks by traversal position (earlier = better)
            chunk_positions = {}
            for step in trace.steps:
                for chunk_id in step.chunk_ids:
                    if chunk_id not in chunk_positions:
                        chunk_positions[chunk_id] = step.step_number

            for chunk_id in graph_chunks:
                position = chunk_positions.get(chunk_id, len(trace.steps))
                # Decay score by position
                graph_score = self.graph_weight * (1.0 / (1.0 + position * 0.1))

                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id] += graph_score
                    chunk_sources[chunk_id] = "both"
                else:
                    chunk_scores[chunk_id] = graph_score
                    chunk_sources[chunk_id] = "graph"

        # Sort by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c[0] for c in sorted_chunks[:max_chunks]]

        return FusedResult(
            chunk_ids=top_chunk_ids,
            chunk_scores=chunk_scores,
            chunk_sources=chunk_sources,
            trace=trace,
        )

    def get_context_chunks(
        self,
        fused: FusedResult,
        limit: int = 12,
    ) -> list[tuple[str, str]]:
        """
        Fetch actual chunk content for LLM context.
        
        Args:
            fused: FusedResult from fusion
            limit: Max chunks to send to LLM
        
        Returns:
            List of (chunk_id, content) tuples
        """
        top_ids = fused.chunk_ids[:limit]
        return self.storage.get_chunk_texts(top_ids)
