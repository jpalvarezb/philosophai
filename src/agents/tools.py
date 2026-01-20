"""Tools available to the multi-hop agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
    from ..graph import GraphBuilder
    from ..rag import VectorSearch


@dataclass
class ToolResult:
    """Result from an agent tool call."""
    tool_name: str
    success: bool
    data: Any
    message: str = ""


class AgentTools:
    """Tools that the multi-hop agent can invoke."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        graph_builder: "GraphBuilder",
        vector_search: "VectorSearch",
        node_to_community: dict[str, int],
    ):
        self.storage = storage
        self.graph = graph_builder.graph
        self.graph_builder = graph_builder
        self.vector_search = vector_search
        self.node_to_community = node_to_community

    def search_vectors(self, query: str, limit: int = 10) -> ToolResult:
        """Search for relevant chunks via vector similarity."""
        try:
            chunk_ids, scores, _ = self.vector_search.search_chunks_only(query, limit=limit)
            return ToolResult(
                tool_name="search_vectors",
                success=True,
                data={
                    "chunk_ids": chunk_ids,
                    "scores": scores,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_vectors",
                success=False,
                data=None,
                message=str(e),
            )

    def read_community_summary(self, community_id: int) -> ToolResult:
        """Read the summary of a community."""
        try:
            df = self.storage.get_communities()
            row = df[df["community_id"] == community_id]
            if row.empty:
                return ToolResult(
                    tool_name="read_community_summary",
                    success=False,
                    data=None,
                    message=f"Community {community_id} not found",
                )
            row = row.iloc[0]
            return ToolResult(
                tool_name="read_community_summary",
                success=True,
                data={
                    "community_id": community_id,
                    "size": row["size"],
                    "top_terms": row["top_terms"],
                    "summary": row["summary"],
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="read_community_summary",
                success=False,
                data=None,
                message=str(e),
            )

    def expand_node(self, node_id: str, max_neighbors: int = 10) -> ToolResult:
        """Get neighbors of a node with edge information."""
        try:
            if self.graph is None or node_id not in self.graph:
                return ToolResult(
                    tool_name="expand_node",
                    success=False,
                    data=None,
                    message=f"Node {node_id} not found in graph",
                )

            neighbors = self.graph_builder.get_neighbors(node_id)[:max_neighbors]
            node_label = self.graph_builder.get_node_label(node_id)
            node_community = self.node_to_community.get(node_id)

            neighbor_data = []
            for neighbor_id, predicate, attrs in neighbors:
                neighbor_data.append({
                    "node_id": neighbor_id,
                    "label": self.graph_builder.get_node_label(neighbor_id),
                    "predicate": attrs.get("label", predicate),
                    "weight": attrs.get("weight", 1),
                    "community_id": self.node_to_community.get(neighbor_id),
                    "chunk_count": len(attrs.get("chunks", [])),
                })

            return ToolResult(
                tool_name="expand_node",
                success=True,
                data={
                    "node_id": node_id,
                    "label": node_label,
                    "community_id": node_community,
                    "neighbors": neighbor_data,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="expand_node",
                success=False,
                data=None,
                message=str(e),
            )

    def get_chunk_content(self, chunk_ids: list[str]) -> ToolResult:
        """Retrieve actual text content for chunks."""
        try:
            chunks = self.storage.get_chunk_texts(chunk_ids)
            return ToolResult(
                tool_name="get_chunk_content",
                success=True,
                data={
                    "chunks": [{"id": c[0], "content": c[1]} for c in chunks],
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="get_chunk_content",
                success=False,
                data=None,
                message=str(e),
            )

    def get_entities_from_chunks(self, chunk_ids: list[str]) -> ToolResult:
        """Find entity IDs mentioned in given chunks."""
        try:
            entity_ids = self.storage.get_entity_ids_from_chunks(chunk_ids)
            return ToolResult(
                tool_name="get_entities_from_chunks",
                success=True,
                data={
                    "entity_ids": entity_ids,
                    "count": len(entity_ids),
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="get_entities_from_chunks",
                success=False,
                data=None,
                message=str(e),
            )

    def sequential_thinking(
        self,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        next_thought_needed: bool,
        is_revision: bool = False,
        revises_thought: int | None = None,
        branch_from_thought: int | None = None,
        branch_id: str | None = None,
        needs_more_thoughts: bool = False,
    ) -> ToolResult:
        """
        Record a step in the sequential thinking process.
        
        This tool allows the agent to structure its reasoning, revise previous thoughts,
        and branch out into different lines of inquiry.
        """
        # Adjust total thoughts if we exceeded the estimate
        if thought_number > total_thoughts:
            total_thoughts = thought_number
            
        return ToolResult(
            tool_name="sequential_thinking",
            success=True,
            data={
                "thought": thought,
                "thought_number": thought_number,
                "total_thoughts": total_thoughts,
                "next_thought_needed": next_thought_needed or needs_more_thoughts,
                "is_revision": is_revision,
                "revises_thought": revises_thought,
                "branch_from_thought": branch_from_thought,
                "branch_id": branch_id,
            },
            message=f"Thought {thought_number}/{total_thoughts} recorded.",
        )
