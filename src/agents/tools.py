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
        self._current_scope: "Scope | None" = None
        self._scoped_text_ids: list[int] = []
        self._scoped_chunks: set[str] | None = None
        self._scoped_edges: set[tuple[str, str, str]] | None = None
        self._induced_nodes: set[str] | None = None  # V(scoped_edges) for induced subgraph
    
    def set_active_scope(self, scope: "Scope | None") -> None:
        """Set the active scope for filtering searches and graph traversal."""
        from .scope import ScopeFilter
        self._current_scope = scope
        if scope and not scope.is_empty():
            scope_filter = ScopeFilter(self.storage, scope)
            self._scoped_text_ids = list(scope_filter.get_text_ids())
            self._scoped_chunks = scope_filter.get_scoped_chunk_ids()
            self._scoped_edges = scope_filter.get_scoped_edges()
            # Compute induced node set: V(scoped_edges)
            self._induced_nodes = set()
            for (subj, obj, _) in self._scoped_edges:
                self._induced_nodes.add(subj)
                self._induced_nodes.add(obj)
        else:
            self._scoped_text_ids = []
            self._scoped_chunks = None
            self._scoped_edges = None
            self._induced_nodes = None
    
    def clear_active_scope(self) -> None:
        """Clear the active scope."""
        self._current_scope = None
        self._scoped_text_ids = []
        self._scoped_chunks = None
        self._scoped_edges = None
        self._induced_nodes = None

    def search_vectors(self, query: str, limit: int = 10) -> ToolResult:
        """Search for relevant chunks via vector similarity, respecting active scope.

        This tool only returns KG-grounded chunks (chunks that have at least one
        triple in normalized_triples_clean_canon), so citations can map back to
        node IDs for UI hover/traceability.
        """
        try:
            requested_limit = max(int(limit), 1)
            # Over-fetch to avoid returning too few chunks after KG-grounding filter.
            probe_limit = requested_limit * 5

            # Use scoped search if scope is active
            if self._scoped_text_ids:
                chunk_ids, scores, _ = self.vector_search.search_chunks_only(
                    query, limit=probe_limit, text_ids=self._scoped_text_ids
                )
            else:
                chunk_ids, scores, _ = self.vector_search.search_chunks_only(query, limit=probe_limit)

            grounded = self.storage.get_chunk_ids_with_triples(chunk_ids)
            filtered_chunk_ids = []
            filtered_scores = []
            for cid, score in zip(chunk_ids, scores):
                if cid in grounded:
                    filtered_chunk_ids.append(cid)
                    filtered_scores.append(score)

            # Trim back down to the requested limit
            filtered_chunk_ids = filtered_chunk_ids[:requested_limit]
            filtered_scores = filtered_scores[:requested_limit]

            return ToolResult(
                tool_name="search_vectors",
                success=True,
                data={
                    "chunk_ids": filtered_chunk_ids,
                    "scores": filtered_scores,
                    "scoped": bool(self._scoped_text_ids),
                    "kg_grounded": True,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_vectors",
                success=False,
                data=None,
                message=str(e),
            )

    def search_community_reports(self, query: str, limit: int = 5) -> ToolResult:
        """
        Search community report summaries to identify relevant topic clusters.

        Use for UNSCOPED/global questions to route into the right communities before
        chunk-level search. When scope is active, cited chunks are pre-filtered by scope.

        This tool only returns KG-grounded cited chunks (those with at least one
        triple in normalized_triples_clean_canon) so downstream citations can map
        back to node IDs.
        """
        try:
            result = self.vector_search.search_community_reports(
                query,
                limit=limit,
                text_ids=self._scoped_text_ids if self._scoped_text_ids else None,
            )

            cited_chunk_ids = result.cited_chunk_ids
            grounded = self.storage.get_chunk_ids_with_triples(cited_chunk_ids)
            cited_chunk_ids = [cid for cid in cited_chunk_ids if cid in grounded]

            return ToolResult(
                tool_name="search_community_reports",
                success=True,
                data={
                    "community_ids": result.community_ids,
                    "scores": result.scores,
                    "cited_chunk_ids": cited_chunk_ids,
                    "scoped": bool(self._scoped_text_ids),
                    "kg_grounded": True,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_community_reports",
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
        """
        Get neighbors of a node with edge information.
        
        If scope is active, edges are marked as "in_scope" if they have
        supporting chunks within the scope. Out-of-scope edges are shown
        but marked clearly, allowing the agent to decide whether to follow them.
        """
        try:
            if self.graph is None or node_id not in self.graph:
                return ToolResult(
                    tool_name="expand_node",
                    success=False,
                    data=None,
                    message=f"Node {node_id} not found in graph",
                )

            neighbors = self.graph_builder.get_neighbors(node_id)
            node_label = self.graph_builder.get_node_label(node_id)
            node_community = self.node_to_community.get(node_id)

            neighbor_data = []
            in_scope_count = 0
            out_scope_count = 0
            
            for neighbor_id, predicate_canon_id, attrs in neighbors:
                edge_label = attrs.get("label", predicate_canon_id)  # Human-readable
                edge_chunks = attrs.get("chunks", [])
                
                # Check if edge is in scope
                # scoped_edges contains (subject_canon_id, object_canon_id, predicate_canon_id)
                # Use predicate_canon_id (the key from MultiDiGraph), NOT the label
                in_scope = True  # Default to in-scope if no scope set
                if self._scoped_edges is not None:
                    # Edge is in scope if (subj, obj, pred) or (obj, subj, pred) in scoped_edges
                    in_scope = (
                        (node_id, neighbor_id, predicate_canon_id) in self._scoped_edges or
                        (neighbor_id, node_id, predicate_canon_id) in self._scoped_edges
                    )
                
                if in_scope:
                    in_scope_count += 1
                else:
                    out_scope_count += 1
                
                # Also check how many chunks are in scope
                scoped_chunk_count = len(edge_chunks)
                if self._scoped_chunks is not None:
                    scoped_chunk_count = len(set(edge_chunks) & self._scoped_chunks)
                
                # Compute traversal score (like GraphTraverser._score_node)
                edge_weight = attrs.get("weight", 1)
                neighbor_label = self.graph_builder.get_node_label(neighbor_id)
                
                # Simplified scoring similar to traversal
                traversal_score = 0.0
                score_breakdown = []
                
                # Edge weight contribution (normalized)
                weight_score = min(edge_weight / 10.0, 1.0) * 0.2
                traversal_score += weight_score
                score_breakdown.append(f"edge_wt={weight_score:.2f}")
                
                # Chunk evidence (more chunks = more confidence)
                chunk_score = min(scoped_chunk_count / 5.0, 1.0) * 0.3 if scoped_chunk_count > 0 else 0
                traversal_score += chunk_score
                score_breakdown.append(f"evidence={chunk_score:.2f}")
                
                # Scope bonus
                if in_scope:
                    traversal_score += 0.3
                    score_breakdown.append("in_scope=+0.30")
                
                neighbor_data.append({
                    "node_id": neighbor_id,
                    "label": neighbor_label,
                    "predicate": edge_label,
                    "weight": edge_weight,
                    "community_id": self.node_to_community.get(neighbor_id),
                    "chunk_count": len(edge_chunks),
                    "in_scope": in_scope,
                    "scoped_chunk_count": scoped_chunk_count,
                    "traversal_score": round(traversal_score, 3),
                    "score_breakdown": "|".join(score_breakdown),
                })
            
            # Sort: in-scope first, then traversal_score, then weight
            neighbor_data.sort(key=lambda x: (-int(x["in_scope"]), -x.get("traversal_score", 0), -x.get("weight", 0)))
            neighbor_data = neighbor_data[:max_neighbors]

            # Check if this node itself is in the induced subgraph
            node_in_scope = True  # Default if no scope
            if self._induced_nodes is not None:
                node_in_scope = node_id in self._induced_nodes
            
            return ToolResult(
                tool_name="expand_node",
                success=True,
                data={
                    "node_id": node_id,
                    "label": node_label,
                    "community_id": node_community,
                    "node_in_scope": node_in_scope,  # Is this node in the induced subgraph?
                    "neighbors": neighbor_data,
                    "total_neighbors": len(neighbors),
                    "in_scope_neighbors": in_scope_count,
                    "out_scope_neighbors": out_scope_count,
                    "scope_active": self._scoped_edges is not None,
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

    def get_entities_from_chunks(self, chunk_ids: list[str], query: str = "") -> ToolResult:
        """
        Find entity IDs mentioned in given chunks, with optional scoring.
        
        If query is provided, entities are scored by:
        - Query relevance (label token overlap)
        - In-scope degree (how many in-scope edges - entities with 0 are filtered)
        - Conceptness score (penalizes multiword phrase artifacts)
        """
        try:
            entity_ids = self.storage.get_entity_ids_from_chunks(chunk_ids)
            
            # If no query, just return raw IDs
            if not query:
                return ToolResult(
                    tool_name="get_entities_from_chunks",
                    success=True,
                    data={
                        "entity_ids": entity_ids,
                        "count": len(entity_ids),
                    },
                )
            
            # Score entities by query relevance
            from ..rag.seeds import tokenize_query, score_label_relevance
            
            query_tokens = tokenize_query(query)
            scored_entities = []
            
            # Use pre-computed induced node set from scope
            induced_nodes = self._induced_nodes
            
            for entity_id in entity_ids:
                if entity_id not in self.graph:
                    continue
                
                label = self.graph.nodes[entity_id].get("label", entity_id)
                relevance, reason = score_label_relevance(label, query_tokens)
                
                # Compute in-scope degree
                in_scope_degree = 0
                if induced_nodes is not None:
                    # Only count if this entity is in the induced subgraph
                    if entity_id in induced_nodes:
                        # Count in-scope edges for this node
                        for neighbor in self.graph[entity_id]:
                            for pred_key in self.graph[entity_id][neighbor]:
                                if ((entity_id, neighbor, pred_key) in self._scoped_edges or
                                    (neighbor, entity_id, pred_key) in self._scoped_edges):
                                    in_scope_degree += 1
                else:
                    # No scope - use total degree
                    in_scope_degree = self.graph.degree(entity_id)
                
                # Skip entities with 0 in-scope edges (not traversable in scope)
                if induced_nodes is not None and in_scope_degree == 0:
                    continue
                
                # Penalize multiword phrase entities (likely extraction artifacts)
                # E.g., "soul body union" should score lower than "soul"
                word_count = len(label.split())
                phrase_penalty = 0.0
                phrase_reason = ""
                if word_count >= 3:
                    phrase_penalty = 0.4  # Strong penalty for 3+ word phrases
                    phrase_reason = f"phrase_penalty({word_count}w)=-0.4"
                elif word_count == 2:
                    phrase_penalty = 0.1  # Mild penalty for 2-word entities
                    phrase_reason = f"phrase_penalty(2w)=-0.1"
                
                # Boost for connectivity (normalized)
                degree_bonus = min(in_scope_degree / 20.0, 0.3)  # Up to 0.3 bonus
                
                # Final score
                final_score = relevance - phrase_penalty + degree_bonus
                
                # Build reason string
                reason_parts = [reason] if reason else []
                if phrase_reason:
                    reason_parts.append(phrase_reason)
                reason_parts.append(f"degree={in_scope_degree}")
                
                # Community info
                comm_id = self.node_to_community.get(entity_id)
                
                scored_entities.append({
                    "entity_id": entity_id,
                    "label": label,
                    "query_relevance": round(final_score, 3),
                    "reason": "|".join(reason_parts),
                    "community_id": comm_id,
                    "in_scope_degree": in_scope_degree,
                })
            
            # Sort by final score
            scored_entities.sort(key=lambda x: x["query_relevance"], reverse=True)
            
            return ToolResult(
                tool_name="get_entities_from_chunks",
                success=True,
                data={
                    "entities": scored_entities[:50],  # Cap at 50
                    "count": len(scored_entities),
                    "query_tokens": list(query_tokens),
                    "scope_active": induced_nodes is not None,
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
