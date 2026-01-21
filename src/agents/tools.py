"""Tools available to the multi-hop agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
    from ..graph import GraphBuilder
    from ..rag import VectorSearch
    from .scope import Scope, ScopeFilter


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

    def set_scope(
        self,
        authors: list[str] | None = None,
        titles: list[str] | None = None,
        traditions: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> ToolResult:
        """
        Restrict retrieval to specific authors, works, traditions, or domains.
        
        Use this when the query asks specifically about one philosopher's view,
        a particular text, or a specific tradition. Do NOT use when:
        - Query asks about influence/reception (need commentators)
        - Query compares multiple traditions broadly
        - Query is open-ended ("what do philosophers think about X")
        
        Args:
            authors: List of author names (e.g., ["Aristotle", "Plato"])
            titles: List of work titles (e.g., ["Republic", "On the Soul (De Anima)"])
            traditions: List of traditions (e.g., ["Greek–Hellenistic", "Indian"])
            domains: List of domains (e.g., ["Ethics", "Metaphysics"])
        
        Valid values:
            authors: Aristotle, Plato, Augustine, Immanuel Kant, David Hume, Seneca,
                     Confucius, Laozi, Vyasa, Shankara, Nagarjuna, Al-Ghazali, Maimonides, etc.
            titles: Republic, Nicomachean Ethics, On the Soul (De Anima), Metaphysics,
                    Critique of Pure Reason, Confessions, Bhagavad Gita, Dao De Jing, etc.
            traditions: Greek–Hellenistic, Indian, Chinese, Japanese, Modern European,
                        Christian, Islamic, Jewish
            domains: Ethics, Metaphysics, Epistemology, Theology, Anthropology, History
        
        Returns:
            Scope object with chunk count in scope
        """
        from .scope import Scope, ScopeFilter, VALID_AUTHORS, VALID_TRADITIONS, VALID_DOMAINS
        
        # Validate inputs
        invalid_authors = [a for a in (authors or []) if a not in VALID_AUTHORS]
        invalid_traditions = [t for t in (traditions or []) if t not in VALID_TRADITIONS]
        invalid_domains = [d for d in (domains or []) if d not in VALID_DOMAINS]
        
        warnings = []
        if invalid_authors:
            warnings.append(f"Unknown authors: {invalid_authors}")
        if invalid_traditions:
            warnings.append(f"Unknown traditions: {invalid_traditions}")
        if invalid_domains:
            warnings.append(f"Unknown domains: {invalid_domains}")
        
        # Create scope with valid values only
        scope = Scope(
            authors=[a for a in (authors or []) if a in VALID_AUTHORS],
            titles=titles or [],  # Titles are flexible, don't validate strictly
            traditions=[t for t in (traditions or []) if t in VALID_TRADITIONS],
            domains=[d for d in (domains or []) if d in VALID_DOMAINS],
            strict=True,
        )
        
        if scope.is_empty():
            return ToolResult(
                tool_name="set_scope",
                success=False,
                data=None,
                message="No valid scope constraints provided. " + "; ".join(warnings) if warnings else "",
            )
        
        # Calculate chunk count
        scope_filter = ScopeFilter(self.storage, scope)
        chunk_count = scope_filter.get_scoped_chunk_count()
        text_ids = scope_filter.get_text_ids()
        
        return ToolResult(
            tool_name="set_scope",
            success=True,
            data={
                "scope": scope.to_dict(),
                "chunk_count": chunk_count,
                "text_count": len(text_ids),
                "description": scope.describe(),
            },
            message=f"Scope set: {scope.describe()}. {chunk_count:,} chunks from {len(text_ids)} texts available." + 
                    (f" Warnings: {'; '.join(warnings)}" if warnings else ""),
        )
    
    def list_available_sources(self, category: str = "authors") -> ToolResult:
        """
        List available sources for scoping.
        
        Args:
            category: One of "authors", "titles", "traditions", "domains"
        
        Returns:
            List of valid values for the category with chunk counts
        """
        try:
            if category == "authors":
                rows = self.storage.con.execute("""
                    SELECT f.author_source, COUNT(DISTINCT c.chunk_id) as chunks
                    FROM files f
                    JOIN chunks c ON f.text_id = c.text_id
                    GROUP BY f.author_source
                    ORDER BY chunks DESC
                """).fetchall()
                data = [{"name": r[0], "chunks": r[1]} for r in rows]
            
            elif category == "titles":
                rows = self.storage.con.execute("""
                    SELECT f.title, f.author_source, COUNT(DISTINCT c.chunk_id) as chunks
                    FROM files f
                    JOIN chunks c ON f.text_id = c.text_id
                    GROUP BY f.title, f.author_source
                    ORDER BY f.author_source, f.title
                """).fetchall()
                data = [{"title": r[0], "author": r[1], "chunks": r[2]} for r in rows]
            
            elif category == "traditions":
                rows = self.storage.con.execute("""
                    SELECT f.tradition, COUNT(DISTINCT c.chunk_id) as chunks
                    FROM files f
                    JOIN chunks c ON f.text_id = c.text_id
                    GROUP BY f.tradition
                    ORDER BY chunks DESC
                """).fetchall()
                data = [{"name": r[0], "chunks": r[1]} for r in rows]
            
            elif category == "domains":
                # Domains are semicolon-separated
                rows = self.storage.con.execute("""
                    SELECT TRIM(domain) as domain, COUNT(*) as file_count
                    FROM (
                        SELECT UNNEST(string_split(domains, ';')) as domain
                        FROM files
                    )
                    GROUP BY 1
                    ORDER BY file_count DESC
                """).fetchall()
                data = [{"name": r[0], "files": r[1]} for r in rows]
            
            else:
                return ToolResult(
                    tool_name="list_available_sources",
                    success=False,
                    data=None,
                    message=f"Unknown category: {category}. Use 'authors', 'titles', 'traditions', or 'domains'.",
                )
            
            return ToolResult(
                tool_name="list_available_sources",
                success=True,
                data={"category": category, "items": data},
                message=f"Found {len(data)} {category}.",
            )
        except Exception as e:
            return ToolResult(
                tool_name="list_available_sources",
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
