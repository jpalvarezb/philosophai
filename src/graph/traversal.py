"""Community-gated best-first graph traversal for multi-hop reasoning."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from ..schema import TraversalStep, TraversalTrace
    from .filters import GraphFilters


@dataclass
class ScoredNode:
    """Node with traversal priority score (for heap)."""
    score: float
    node_id: str
    from_node_id: str | None
    edge_label: str | None
    depth: int
    community_id: int | None

    def __lt__(self, other: "ScoredNode"):
        # Higher score = higher priority (use negative for min-heap)
        return self.score > other.score


class GraphTraverser:
    """
    Best-first traversal that respects community boundaries.
    
    Prioritizes:
    1. Query relevance (label token overlap)
    2. Nodes within target communities
    3. High-weight edges with valuable predicates
    4. Depth penalty (prefer shallower paths)
    """

    def __init__(
        self,
        graph: "nx.MultiDiGraph",
        node_to_community: dict[str, int],
        filters: "GraphFilters | None" = None,
        # Scoring weights
        query_relevance_weight: float = 0.4,
        edge_weight_factor: float = 0.2,
        community_affinity: float = 0.3,
        depth_penalty: float = 0.05,
    ):
        self.graph = graph
        self.node_to_community = node_to_community
        self.filters = filters
        self.query_relevance_weight = query_relevance_weight
        self.edge_weight_factor = edge_weight_factor
        self.community_affinity = community_affinity
        self.depth_penalty = depth_penalty
        self._query_tokens: set[str] = set()  # Set per-traversal

    def traverse(
        self,
        seed_nodes: list[str],
        target_communities: list[int],
        query: str = "",
        max_hops: int = 2,
        max_nodes: int = 50,
        seed_cap: int = 20,
        beam_width: int = 25,
        max_collected_chunks: int = 120,
        restrict_to_communities: bool = False,
        scoped_chunks: set[str] | None = None,
        scoped_edges: set[tuple[str, str, str]] | None = None,
    ) -> "TraversalTrace":
        """
        Perform best-first traversal from seed nodes.
        
        Args:
            seed_nodes: Starting node IDs (already filtered/scored by caller)
            target_communities: Community IDs to prioritize
            max_hops: Maximum depth from any seed
            max_nodes: Stop after visiting this many expansion nodes
            seed_cap: Maximum seeds to use (prevents budget exhaustion)
            beam_width: Max nodes to expand per depth level
            max_collected_chunks: Stop traversal after collecting this many chunks
            restrict_to_communities: If True, strictly stay within target communities
            scoped_chunks: If provided, only collect chunks in this set (scope filter)
            scoped_edges: If provided, only traverse edges in this set (strict scope).
                          Format: set of (subject_id, object_id, predicate_id) tuples.
                          Entity scope is derived as V(scoped_edges).
        
        Returns:
            TraversalTrace with visited nodes, edges, and collected chunks
        """
        from ..schema import TraversalStep, TraversalTrace
        import re

        trace = TraversalTrace(
            query=query,
            seed_nodes=seed_nodes[:seed_cap],
            seed_communities=target_communities,
        )

        # Extract query tokens for relevance scoring during expansion
        if query:
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how',
                         'why', 'when', 'where', 'which', 'who', 'to', 'of', 'in',
                         'for', 'on', 'with', 'between', 'relationship'}
            self._query_tokens = {t for t in re.findall(r'[a-z]+', query.lower())
                                  if len(t) > 2 and t not in stopwords}
        else:
            self._query_tokens = set()

        target_set = set(target_communities)
        visited: set[str] = set()
        heap: list[ScoredNode] = []
        
        # Track expansion count separately from seed processing
        expansion_count = 0
        
        # Diagnostic counters (instance vars for access in _expand_node)
        self._filtered_low_quality = 0
        self._filtered_blocked_pred = 0
        self._filtered_community = 0
        self._filtered_out_of_scope_edges = 0  # Edges skipped due to scope
        self._stopped_by_chunk_cap = False
        self._scoped_chunks = scoped_chunks  # For _get_edge_chunks filtering
        self._scoped_edges = scoped_edges  # For edge-level scope filtering
        
        # Derive scoped_entity_ids from scoped_edges (V(E))
        # This is a cache, not a separate authority
        self._scoped_entity_ids: set[str] | None = None
        if scoped_edges is not None:
            self._scoped_entity_ids = set()
            for subj, obj, _ in scoped_edges:
                self._scoped_entity_ids.add(subj)
                self._scoped_entity_ids.add(obj)

        # Initialize heap with capped seeds (seeds are depth=0)
        seeds_added = 0
        for node_id in seed_nodes:
            if seeds_added >= seed_cap:
                break
            if node_id not in self.graph:
                continue
            
            # In scoped mode, skip seeds not in V(scoped_edges)
            if self._scoped_entity_ids is not None and node_id not in self._scoped_entity_ids:
                continue
            
            community_id = self.node_to_community.get(node_id)
            
            # In strict mode, skip seeds not in target communities
            if restrict_to_communities and target_set and community_id not in target_set:
                continue
            
            score = self._score_node(node_id, None, target_set, 0)
            heapq.heappush(
                heap,
                ScoredNode(
                    score=score,
                    node_id=node_id,
                    from_node_id=None,
                    edge_label=None,
                    depth=0,
                    community_id=community_id,
                ),
            )
            seeds_added += 1

        step_number = 0
        nodes_at_depth: dict[int, int] = {}  # Track beam width per depth

        while heap and expansion_count < max_nodes:
            # Check chunk collection cap
            if len(trace.collected_chunk_ids) >= max_collected_chunks:
                self._stopped_by_chunk_cap = True
                break
            
            current = heapq.heappop(heap)

            if current.node_id in visited:
                continue
            
            # Beam width check: limit nodes processed per depth
            depth = current.depth
            nodes_at_depth[depth] = nodes_at_depth.get(depth, 0) + 1
            if depth > 0 and nodes_at_depth[depth] > beam_width:
                continue

            # Filter check during expansion (less strict than seeding)
            if self.filters and depth > 0:
                if not self.filters.is_valid_expansion(current.node_id):
                    continue

            visited.add(current.node_id)
            
            # Count toward budget only for expansions (depth > 0)
            if depth > 0:
                expansion_count += 1

            # Collect chunks from edge used to reach this node
            edge_chunks = []
            if current.from_node_id:
                edge_chunks = self._get_edge_chunks(
                    current.from_node_id, current.node_id, current.edge_label
                )

            # Record step (seeds at depth=0, expansions at depth>=1)
            step = TraversalStep(
                step_number=step_number,
                node_id=current.node_id,
                node_label=self._get_label(current.node_id),
                edge_label=current.edge_label,
                from_node_id=current.from_node_id,
                community_id=current.community_id,
                depth=current.depth,
                chunk_ids=edge_chunks,
                score=current.score,
            )
            trace.add_step(step)
            step_number += 1

            # Expand neighbors if within depth limit
            if current.depth < max_hops:
                self._expand_node(
                    current.node_id,
                    current.depth,
                    target_communities,
                    visited,
                    heap,
                    restrict_to_communities,
                )

        # Store diagnostic counters in trace
        trace.filtered_low_quality = self._filtered_low_quality
        trace.filtered_blocked_pred = self._filtered_blocked_pred
        trace.filtered_community = self._filtered_community
        trace.filtered_out_of_scope_edges = self._filtered_out_of_scope_edges
        trace.stopped_by_chunk_cap = self._stopped_by_chunk_cap
        
        return trace

    def _score_node(
        self,
        node_id: str,
        edge_weight: int | None,
        target_communities: set[int],
        depth: int,
        predicate: str | None = None,
    ) -> float:
        """Calculate priority score for a node."""
        import re
        score = 0.0

        # Query relevance: check label token overlap
        if self._query_tokens:
            label = self.graph.nodes.get(node_id, {}).get("label", node_id)
            label_tokens = set(re.findall(r'[a-z]+', label.lower()))
            overlap = label_tokens & self._query_tokens
            if overlap:
                # Proportional to overlap
                relevance = len(overlap) / max(len(self._query_tokens), 1)
                score += self.query_relevance_weight * relevance

        # Community affinity bonus
        community_id = self.node_to_community.get(node_id)
        if community_id in target_communities:
            score += self.community_affinity

        # Edge weight bonus (normalized), with predicate quality
        if edge_weight:
            weight_score = self.edge_weight_factor * min(edge_weight / 10.0, 1.0)
            if self.filters and predicate:
                pred_weight = self.filters.predicate_weight(predicate)
                if pred_weight == 0:  # Blocked predicate
                    return -1.0  # Signal to skip this edge
                weight_score *= pred_weight
            score += weight_score

        # Depth penalty (prefer shallower)
        score -= self.depth_penalty * depth

        return score

    def _expand_node(
        self,
        node_id: str,
        current_depth: int,
        target_communities: set[int],
        visited: set[str],
        heap: list[ScoredNode],
        restrict_to_communities: bool = False,
    ):
        """Add neighbors of node to the heap."""
        if node_id not in self.graph:
            return

        for neighbor_id in self.graph[node_id]:
            if neighbor_id in visited:
                continue
            
            # Scoped entity filter: skip neighbors not in scoped entity set
            if self._scoped_entity_ids is not None and neighbor_id not in self._scoped_entity_ids:
                self._filtered_out_of_scope_edges += 1
                continue
            
            # Filter check: skip stop entities during expansion
            if self.filters and not self.filters.is_valid_expansion(neighbor_id):
                self._filtered_low_quality += 1
                continue

            neighbor_community = self.node_to_community.get(neighbor_id)
            
            # In strict mode, only expand to nodes within target communities
            if restrict_to_communities and target_communities and neighbor_community not in target_communities:
                self._filtered_community += 1
                continue

            # Get best IN-SCOPE edge to this neighbor (by weight, with predicate quality)
            edges = self.graph[node_id][neighbor_id]
            best_edge_key = None
            best_score = -1
            
            for key in edges:
                # Check if this specific edge is in scope
                if self._scoped_edges is not None:
                    # Edge must be in scoped_edges set (check both directions for undirected semantics)
                    edge_in_scope = (
                        (node_id, neighbor_id, key) in self._scoped_edges or
                        (neighbor_id, node_id, key) in self._scoped_edges
                    )
                    if not edge_in_scope:
                        continue  # Skip this edge, try others
                
                weight = edges[key].get("weight", 1)
                label = edges[key].get("label", key)
                modifier = self.filters.edge_weight_modifier(label) if self.filters else 1.0
                adjusted = weight * modifier
                if adjusted > best_score:
                    best_score = adjusted
                    best_edge_key = key
            
            if best_edge_key is None:
                # No in-scope edges to this neighbor
                if self._scoped_edges is not None:
                    self._filtered_out_of_scope_edges += 1
                continue
                
            edge_data = edges[best_edge_key]
            edge_weight = edge_data.get("weight", 1)
            edge_label = edge_data.get("label", best_edge_key)

            score = self._score_node(
                neighbor_id, edge_weight, target_communities, 
                current_depth + 1, predicate=edge_label
            )
            
            # Skip if score is negative (blocked predicate)
            if score < 0:
                self._filtered_blocked_pred += 1
                continue

            heapq.heappush(
                heap,
                ScoredNode(
                    score=score,
                    node_id=neighbor_id,
                    from_node_id=node_id,
                    edge_label=edge_label,
                    depth=current_depth + 1,
                    community_id=neighbor_community,
                ),
            )

    def _get_edge_chunks(self, from_id: str, to_id: str, predicate: str | None) -> list[str]:
        """
        Get chunk IDs from an edge.
        
        If scoped_chunk_ids was provided to traverse(), only returns chunks in that set.
        """
        if from_id not in self.graph or to_id not in self.graph[from_id]:
            return []
        edges = self.graph[from_id][to_id]
        all_chunks = []
        for key, data in edges.items():
            if predicate is None or key == predicate or data.get("label") == predicate:
                all_chunks.extend(data.get("chunks", []))
        unique_chunks = list(set(all_chunks))
        
        # Apply scope filter if set
        if self._scoped_chunks is not None:
            return [c for c in unique_chunks if c in self._scoped_chunks]
        return unique_chunks

    def _get_label(self, node_id: str) -> str:
        """Get human-readable label for a node."""
        return self.graph.nodes.get(node_id, {}).get("label", node_id)
