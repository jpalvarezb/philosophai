"""Community-gated best-first graph traversal for multi-hop reasoning."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from ..schema import TraversalStep, TraversalTrace


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
    1. Nodes within target communities (from vector search on summaries)
    2. High-weight edges
    3. Semantic similarity to query (if embeddings available)
    """

    def __init__(
        self,
        graph: "nx.MultiDiGraph",
        node_to_community: dict[str, int],
        query_embedding: list[float] | None = None,
        # Scoring weights
        semantic_weight: float = 0.5,
        edge_weight_factor: float = 0.2,
        community_affinity: float = 0.3,
    ):
        self.graph = graph
        self.node_to_community = node_to_community
        self.query_embedding = query_embedding
        self.semantic_weight = semantic_weight
        self.edge_weight_factor = edge_weight_factor
        self.community_affinity = community_affinity

    def traverse(
        self,
        seed_nodes: list[str],
        target_communities: list[int],
        max_hops: int = 2,
        max_nodes: int = 50,
        max_cross_community_hops: int = 1,
        restrict_to_communities: bool = False,
    ) -> "TraversalTrace":
        """
        Perform best-first traversal from seed nodes.
        
        Args:
            seed_nodes: Starting node IDs (from vector search)
            target_communities: Community IDs to prioritize (from summary search)
            max_hops: Maximum depth from any seed
            max_nodes: Stop after visiting this many nodes
            max_cross_community_hops: Limit hops outside target communities
            restrict_to_communities: If True, strictly stay within target communities (GraphRAG mode)
        
        Returns:
            TraversalTrace with visited nodes, edges, and collected chunks
        """
        from ..schema import TraversalStep, TraversalTrace

        trace = TraversalTrace(
            query="",  # Set by caller
            seed_nodes=seed_nodes,
            seed_communities=target_communities,
        )

        target_set = set(target_communities)
        visited: set[str] = set()
        heap: list[ScoredNode] = []

        # Initialize heap with seed nodes
        for node_id in seed_nodes:
            if node_id in self.graph:
                community_id = self.node_to_community.get(node_id)
                # In strict mode, skip seeds not in target communities (unless no targets)
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

        step_number = 0
        cross_community_count = 0

        while heap and len(visited) < max_nodes:
            current = heapq.heappop(heap)

            if current.node_id in visited:
                continue

            # Check cross-community limit
            if current.community_id is not None and current.community_id not in target_set:
                if restrict_to_communities:
                    # Strict mode: skip all nodes outside target communities
                    continue
                if cross_community_count >= max_cross_community_hops:
                    continue
                cross_community_count += 1

            visited.add(current.node_id)

            # Collect chunks from edge used to reach this node
            edge_chunks = []
            if current.from_node_id:
                edge_chunks = self._get_edge_chunks(
                    current.from_node_id, current.node_id, current.edge_label
                )

            # Record step
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
                    target_set,
                    visited,
                    heap,
                    restrict_to_communities,
                )

        return trace

    def _score_node(
        self,
        node_id: str,
        edge_weight: int | None,
        target_communities: set[int],
        depth: int,
    ) -> float:
        """Calculate priority score for a node."""
        score = 0.0

        # Community affinity bonus
        community_id = self.node_to_community.get(node_id)
        if community_id in target_communities:
            score += self.community_affinity

        # Edge weight bonus (normalized)
        if edge_weight:
            score += self.edge_weight_factor * min(edge_weight / 10.0, 1.0)

        # Depth penalty (prefer shallower)
        score -= 0.05 * depth

        # TODO: Add semantic similarity if query_embedding and node embeddings available
        # For now, this would require node-level embeddings which we don't have yet

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

            neighbor_community = self.node_to_community.get(neighbor_id)
            
            # In strict mode, only expand to nodes within target communities
            if restrict_to_communities and target_communities and neighbor_community not in target_communities:
                continue

            # Get best edge to this neighbor
            edges = self.graph[node_id][neighbor_id]
            best_edge_key = max(edges.keys(), key=lambda k: edges[k].get("weight", 1))
            edge_data = edges[best_edge_key]
            edge_weight = edge_data.get("weight", 1)
            edge_label = edge_data.get("label", best_edge_key)

            score = self._score_node(neighbor_id, edge_weight, target_communities, current_depth + 1)

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
        """Get chunk IDs from an edge."""
        if from_id not in self.graph or to_id not in self.graph[from_id]:
            return []
        edges = self.graph[from_id][to_id]
        all_chunks = []
        for key, data in edges.items():
            if predicate is None or key == predicate or data.get("label") == predicate:
                all_chunks.extend(data.get("chunks", []))
        return list(set(all_chunks))

    def _get_label(self, node_id: str) -> str:
        """Get human-readable label for a node."""
        return self.graph.nodes.get(node_id, {}).get("label", node_id)
