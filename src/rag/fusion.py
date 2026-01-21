"""Fuse and rerank results from vector search and graph traversal."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from ..config.logging import trace_logger, TRACE_VERBOSE, TRACE_MAX_ITEMS

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
    from ..schema import TraversalTrace
    from .vector import VectorSearchResult, CommunityReportSearchResult


# Default config for context selection
MAX_CONTEXT_CHUNKS = 35
MUST_KEEP_REPORT_CHUNKS = 8
MUST_KEEP_TRAVERSAL_CHUNKS = 12


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
        trace_logger.decision(
            f"fusion_result total={len(chunk_scores)} selected={len(top_chunk_ids)} "
            f"top={top_chunk_ids[:TRACE_MAX_ITEMS]}"
        )

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

    def select_context_chunks(
        self,
        query: str,
        report_cited_chunks: list[str],
        traversal_chunks: list[str],
        traversal_chunk_scores: dict[str, float],
        vector_search_chunks: list[str],
        query_embedding: list[float] | None = None,
        max_context: int = MAX_CONTEXT_CHUNKS,
        must_keep_report: int = MUST_KEEP_REPORT_CHUNKS,
        must_keep_traversal: int = MUST_KEEP_TRAVERSAL_CHUNKS,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Select final context chunks for LLM, with must-keep guarantees.
        
        Priority:
        1. Must-keep: top report-cited chunks
        2. Must-keep: top traversal-scored chunks
        3. Fill: vector search chunks by similarity
        4. Fill: remaining traversal chunks
        
        Args:
            query: Original question
            report_cited_chunks: Chunks cited by community reports
            traversal_chunks: All chunks from traversal
            traversal_chunk_scores: Scores from traversal (step position, etc.)
            vector_search_chunks: Chunks from direct vector search
            query_embedding: Pre-computed query embedding (optional)
            max_context: Maximum chunks for LLM context
            must_keep_report: Guaranteed report chunk slots
            must_keep_traversal: Guaranteed traversal chunk slots
        
        Returns:
            (context_chunk_ids, chunk_scores) - ordered list and score dict
        """
        selected: list[str] = []
        scores: dict[str, float] = {}
        seen: set[str] = set()
        
        def add_chunk(chunk_id: str, score: float, source: str) -> bool:
            """Add chunk if not seen and under limit."""
            if chunk_id in seen or len(selected) >= max_context:
                return False
            selected.append(chunk_id)
            scores[chunk_id] = score
            seen.add(chunk_id)
            return True
        
        # 1. Must-keep: top report-cited chunks
        for i, chunk_id in enumerate(report_cited_chunks[:must_keep_report]):
            score = 1.0 - (i * 0.05)  # Decay by position
            add_chunk(chunk_id, score, "report")
        
        # 2. Must-keep: top traversal chunks by score
        traversal_sorted = sorted(
            [(cid, traversal_chunk_scores.get(cid, 0)) for cid in traversal_chunks],
            key=lambda x: -x[1]
        )
        added_traversal = 0
        for chunk_id, score in traversal_sorted:
            if added_traversal >= must_keep_traversal:
                break
            if add_chunk(chunk_id, score, "traversal"):
                added_traversal += 1
        
        # 3. Fill: vector search chunks (already ranked by similarity)
        for i, chunk_id in enumerate(vector_search_chunks):
            if len(selected) >= max_context:
                break
            score = 0.8 - (i * 0.02)  # Decay by position
            add_chunk(chunk_id, score, "vector")
        
        # 4. Fill: remaining traversal chunks
        for chunk_id, score in traversal_sorted:
            if len(selected) >= max_context:
                break
            add_chunk(chunk_id, score * 0.8, "traversal_fill")
        
        return selected, scores

    def get_context_with_selection(
        self,
        query: str,
        report_cited_chunks: list[str],
        vector_chunks: list[str],
        trace: "TraversalTrace | None",
        max_context: int = MAX_CONTEXT_CHUNKS,
    ) -> tuple[list[tuple[str, str]], list[str], dict[str, float]]:
        """
        High-level method: select context and fetch content.
        
        Args:
            query: Original question
            report_cited_chunks: Chunks from community reports (already scope-filtered)
            vector_chunks: Chunks from vector search (already scope-filtered)
            trace: TraversalTrace with collected chunks (already scope-filtered)
            max_context: Maximum chunks for LLM context
        
        Returns:
            (context_texts, all_collected_ids, context_scores)
            - context_texts: [(chunk_id, content), ...] for LLM
            - all_collected_ids: full set for UI/citations
            - context_scores: scores for selected context
        """
        # Gather all collected chunks (for UI)
        all_collected = set(vector_chunks)
        all_collected.update(report_cited_chunks)
        
        traversal_chunks = []
        traversal_scores = {}
        if trace:
            traversal_chunks = list(set(trace.collected_chunk_ids))
            all_collected.update(traversal_chunks)
            # Score by step position (earlier = higher)
            for step in trace.steps:
                for chunk_id in step.chunk_ids:
                    if chunk_id not in traversal_scores:
                        traversal_scores[chunk_id] = max(0.5, 1.0 - step.step_number * 0.02)
        
        # Select context
        context_ids, context_scores = self.select_context_chunks(
            query=query,
            report_cited_chunks=report_cited_chunks,
            traversal_chunks=traversal_chunks,
            traversal_chunk_scores=traversal_scores,
            vector_search_chunks=vector_chunks,
            max_context=max_context,
        )
        trace_logger.decision(
            f"context_select report={len(report_cited_chunks)} traversal={len(traversal_chunks)} "
            f"vector={len(vector_chunks)} selected={len(context_ids)}"
        )
        if TRACE_VERBOSE:
            trace_logger.debug(f"context_ids_top={context_ids[:TRACE_MAX_ITEMS]}")
        
        # Fetch content
        trace_logger.tool_call(
            "storage.get_chunk_texts",
            count=len(context_ids),
        )
        context_texts = self.storage.get_chunk_texts(context_ids)
        trace_logger.tool_result(
            "storage.get_chunk_texts",
            count=len(context_texts),
        )
        
        return context_texts, list(all_collected), context_scores
