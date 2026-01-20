"""Community detection and summarization using Leiden algorithm."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage
    from ..schema import Community


class CommunityDetector:
    """
    Detects communities in the knowledge graph using Leiden algorithm.
    
    Supports two modes:
    - cluster_graph: Use pre-built cluster projection (recommended)
    - graph: Use full directed graph with filtering (legacy)
    """

    def __init__(
        self,
        storage: "DuckDBStorage",
        graph: nx.MultiDiGraph | None = None,
        cluster_graph: nx.Graph | None = None,
        llm_client: "OpenAI | None" = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Args:
            storage: DuckDB storage instance
            graph: Full directed graph for traversal/reporting (optional)
            cluster_graph: Undirected cluster projection for Leiden (recommended)
            llm_client: OpenAI client for summaries
            embedding_model: Model for embeddings
        """
        self.storage = storage
        self.graph = graph  # Full graph for reporting
        self.cluster_graph = cluster_graph  # Undirected for clustering
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.communities: dict[int, list[str]] = {}  # community_id -> node_ids
        self.node_to_community: dict[str, int] = {}  # node_id -> community_id

    def detect(
        self,
        resolution: float = 1.0,
        min_edge_weight: int = 1,
        max_node_degree_pct: float = 0.01,
        use_weight_transform: bool = True,
    ) -> dict[int, list[str]]:
        """
        Run Leiden community detection.
        
        Uses cluster_graph if provided (recommended), otherwise falls back to
        filtering the full directed graph.

        Args:
            resolution: Higher values = more/smaller communities. Default 1.0.
            min_edge_weight: Drop edges with weight below this. Default 1 (keep all).
            max_node_degree_pct: Drop "stop entities" with degree > this % of total nodes.
            use_weight_transform: Apply log1p(weight) to reduce heavy edge dominance.

        Returns:
            Dict mapping community_id -> list of node_ids (only clustered nodes)
        """
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "leidenalg and python-igraph required. "
                "Install with: pip install leidenalg python-igraph"
            )

        # Use cluster projection if available (recommended)
        if self.cluster_graph is not None:
            return self._detect_from_cluster_graph(
                resolution, min_edge_weight, max_node_degree_pct, use_weight_transform
            )
        elif self.graph is not None:
            return self._detect_from_directed_graph(
                resolution, min_edge_weight, max_node_degree_pct, use_weight_transform
            )
        else:
            raise ValueError("Either cluster_graph or graph must be provided")

    def _detect_from_cluster_graph(
        self,
        resolution: float,
        min_edge_weight: int,
        max_node_degree_pct: float,
        use_weight_transform: bool,
    ) -> dict[int, list[str]]:
        """
        Run Leiden on the cluster projection graph.
        This is the recommended mode - predicates are already collapsed.
        """
        import math
        
        G = self.cluster_graph
        total_nodes = G.number_of_nodes()
        print("🔍 Using cluster projection graph...")
        print(f"   Input: {total_nodes} nodes, {G.number_of_edges()} edges")

        # Filter by edge weight
        filtered_edges = []
        skipped = 0
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            if weight >= min_edge_weight:
                # Apply weight transform to reduce dominance of heavy edges
                transformed_weight = math.log1p(weight) if use_weight_transform else weight
                filtered_edges.append((u, v, transformed_weight))
            else:
                skipped += 1
        
        if min_edge_weight > 1:
            print(f"   Skipped {skipped} edges with weight < {min_edge_weight}")
        if use_weight_transform:
            print(f"   Using log1p weight transform")

        # Build node set
        node_set = set()
        for u, v, _ in filtered_edges:
            node_set.add(u)
            node_set.add(v)

        # Filter stop entities
        if max_node_degree_pct > 0 and node_set:
            degree_threshold = max(1, int(len(node_set) * max_node_degree_pct))
            degree_count = defaultdict(int)
            for u, v, _ in filtered_edges:
                degree_count[u] += 1
                degree_count[v] += 1

            stop_entities = {n for n, d in degree_count.items() if d > degree_threshold}
            if stop_entities:
                print(f"   Dropping {len(stop_entities)} stop entities (degree > {degree_threshold})")
                filtered_edges = [
                    (u, v, w) for u, v, w in filtered_edges
                    if u not in stop_entities and v not in stop_entities
                ]
                node_set -= stop_entities

        print(f"   Clustering: {len(node_set)} nodes, {len(filtered_edges)} edges")

        if not filtered_edges:
            print("⚠️ No edges remain after filtering.")
            return {}

        # Build igraph
        node_list = list(node_set)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v, _ in filtered_edges]
        ig_weights = [w for _, _, w in filtered_edges]

        ig_graph = ig.Graph(n=len(node_list), edges=ig_edges)
        ig_graph.vs["name"] = node_list
        ig_graph.es["weight"] = ig_weights

        print(f"🧮 Running Leiden algorithm (resolution={resolution})...")
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=ig_weights,
        )

        # Build community mappings - ONLY for clustered nodes
        self.communities = defaultdict(list)
        for idx, community_id in enumerate(partition.membership):
            node_id = node_list[idx]
            self.communities[community_id].append(node_id)
            self.node_to_community[node_id] = community_id

        # Unclustered nodes get comm_id = -1 (NOT their own singleton community)
        all_nodes = set(G.nodes())
        unclustered = all_nodes - set(self.node_to_community.keys())
        for node_id in unclustered:
            self.node_to_community[node_id] = -1  # Mark as unclustered

        self._print_stats(total_nodes, len(node_set), unclustered)
        return dict(self.communities)

    def _detect_from_directed_graph(
        self,
        resolution: float,
        min_edge_weight: int,
        max_node_degree_pct: float,
        use_weight_transform: bool,
    ) -> dict[int, list[str]]:
        """
        Legacy mode: Run Leiden on filtered directed graph.
        Not recommended - use cluster projection instead.
        """
        import math
        
        total_nodes = self.graph.number_of_nodes()
        print("🔍 Using directed graph (legacy mode)...")
        print(f"   Original: {total_nodes} nodes, {self.graph.number_of_edges()} edges")

        # Build filtered edge list
        filtered_edges = []
        skipped = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            weight = data.get("weight", 1)
            if weight >= min_edge_weight:
                transformed_weight = math.log1p(weight) if use_weight_transform else weight
                filtered_edges.append((u, v, transformed_weight))
            else:
                skipped += 1
        
        if min_edge_weight > 1:
            print(f"   Skipped {skipped} edges with weight < {min_edge_weight}")

        # Build node set
        node_set = set()
        for u, v, _ in filtered_edges:
            node_set.add(u)
            node_set.add(v)

        # Filter stop entities
        if max_node_degree_pct > 0 and node_set:
            degree_threshold = max(1, int(len(node_set) * max_node_degree_pct))
            degree_count = defaultdict(int)
            for u, v, _ in filtered_edges:
                degree_count[u] += 1
                degree_count[v] += 1

            stop_entities = {n for n, d in degree_count.items() if d > degree_threshold}
            if stop_entities:
                print(f"   Dropping {len(stop_entities)} stop entities (degree > {degree_threshold})")
                filtered_edges = [
                    (u, v, w) for u, v, w in filtered_edges
                    if u not in stop_entities and v not in stop_entities
                ]
                node_set -= stop_entities

        print(f"   Clustering: {len(node_set)} nodes, {len(filtered_edges)} edges")

        if not filtered_edges:
            print("⚠️ No edges remain.")
            return {}

        # Build igraph
        node_list = list(node_set)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v, _ in filtered_edges]
        ig_weights = [w for _, _, w in filtered_edges]

        ig_graph = ig.Graph(n=len(node_list), edges=ig_edges)
        ig_graph.vs["name"] = node_list
        ig_graph.es["weight"] = ig_weights

        print(f"🧮 Running Leiden (resolution={resolution})...")
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=ig_weights,
        )

        # Build community mappings - ONLY for clustered nodes
        self.communities = defaultdict(list)
        for idx, community_id in enumerate(partition.membership):
            node_id = node_list[idx]
            self.communities[community_id].append(node_id)
            self.node_to_community[node_id] = community_id

        # Unclustered nodes get comm_id = -1
        all_nodes = set(self.graph.nodes())
        unclustered = all_nodes - set(self.node_to_community.keys())
        for node_id in unclustered:
            self.node_to_community[node_id] = -1

        self._print_stats(total_nodes, len(node_set), unclustered)
        return dict(self.communities)

    def _print_stats(self, total_nodes: int, clustered_nodes: int, unclustered: set):
        """Print community statistics with coverage metrics."""
        coverage = 100 * clustered_nodes / total_nodes if total_nodes > 0 else 0
        
        print(f"✅ Found {len(self.communities)} communities")
        print(f"   Clustered: {clustered_nodes:,} / {total_nodes:,} nodes ({coverage:.1f}% coverage)")
        print(f"   Unclustered: {len(unclustered):,} nodes (comm_id = -1)")

        sizes = sorted([len(nodes) for nodes in self.communities.values()], reverse=True)
        if sizes:
            print(f"   Largest: {sizes[0]} nodes")
            print(f"   Top 5 sizes: {sizes[:5]}")
            print(f"   Communities with >10 nodes: {sum(1 for s in sizes if s > 10)}")
            print(f"   Communities with >50 nodes: {sum(1 for s in sizes if s > 50)}")
            print(f"   Communities with >100 nodes: {sum(1 for s in sizes if s > 100)}")

    def _get_working_graph(self):
        """Get whichever graph is available, preferring full graph."""
        if self.graph is not None:
            return self.graph
        elif self.cluster_graph is not None:
            return self.cluster_graph
        else:
            raise ValueError("No graph available")

    def get_community_top_terms(self, community_id: int, top_k: int = 10) -> list[str]:
        """Get most representative node labels for a community."""
        if community_id not in self.communities:
            return []

        node_ids = self.communities[community_id]
        G = self._get_working_graph()
        
        # Score by degree within community
        subgraph = G.subgraph(node_ids)
        degree_scores = dict(subgraph.degree())

        # Sort by degree, get labels
        sorted_nodes = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n[0] for n in sorted_nodes[:top_k]]

        labels = []
        for node_id in top_nodes:
            label = G.nodes.get(node_id, {}).get("label", node_id)
            labels.append(label)
        return labels

    def summarize_community(self, community_id: int) -> str:
        """Generate LLM summary for a community based on its top terms and structure."""
        if self.llm_client is None:
            raise ValueError("LLM client required for summarization")

        top_terms = self.get_community_top_terms(community_id, top_k=15)
        node_ids = self.communities.get(community_id, [])
        G = self._get_working_graph()

        # Get sample edges from this community
        sample_triples = []
        subgraph = G.subgraph(node_ids)
        
        for u, v, *rest in list(subgraph.edges(data=True))[:20]:
            data = rest[0] if rest else {}
            u_label = G.nodes.get(u, {}).get("label", u)
            v_label = G.nodes.get(v, {}).get("label", v)
            # For cluster graph, use top_predicates; for full graph, use label
            pred_label = data.get("label") or (data.get("top_predicates", ["related to"])[0] if data.get("top_predicates") else "related to")
            sample_triples.append(f"({u_label}, {pred_label}, {v_label})")

        prompt = f"""Summarize this knowledge graph community in 2-3 sentences.

Community contains {len(node_ids)} entities.

Key terms: {', '.join(top_terms)}

Sample relationships:
{chr(10).join(sample_triples[:15])}

Write a concise summary describing what this community is about - the main themes, concepts, or domain it covers."""

        response = self.llm_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self.llm_client is None:
            raise ValueError("LLM client required for embeddings")
        response = self.llm_client.embeddings.create(
            input=[text], model=self.embedding_model
        )
        return response.data[0].embedding

    def resolution_sweep(
        self,
        resolutions: list[float] | None = None,
        min_edge_weight: int = 1,
        max_node_degree_pct: float = 0.01,
        use_weight_transform: bool = True,
    ) -> dict[float, dict]:
        """
        Run Leiden at multiple resolutions and compare results.

        Args:
            resolutions: List of resolution values to try. Default [0.4, 0.6, 0.8, 1.0, 1.2]
            min_edge_weight: Minimum edge weight filter (default 1 = keep all)
            max_node_degree_pct: Max degree percentage for stop entity filtering
            use_weight_transform: Apply log1p transform to weights

        Returns:
            Dict mapping resolution -> stats dict
        """
        if resolutions is None:
            resolutions = [0.4, 0.6, 0.8, 1.0, 1.2]

        # Get total nodes for coverage calculation
        if self.cluster_graph is not None:
            total_nodes = self.cluster_graph.number_of_nodes()
        elif self.graph is not None:
            total_nodes = self.graph.number_of_nodes()
        else:
            total_nodes = 0

        results = {}
        print("\n" + "=" * 60)
        print("RESOLUTION SWEEP")
        print("=" * 60)

        for res in resolutions:
            print(f"\n--- Resolution {res} ---")
            # Reset state
            self.communities = {}
            self.node_to_community = {}

            self.detect(
                resolution=res,
                min_edge_weight=min_edge_weight,
                max_node_degree_pct=max_node_degree_pct,
                use_weight_transform=use_weight_transform,
            )

            sizes = [len(nodes) for nodes in self.communities.values()]
            clustered = sum(sizes)
            results[res] = {
                "num_communities": len(self.communities),
                "clustered_nodes": clustered,
                "coverage_pct": 100 * clustered / total_nodes if total_nodes > 0 else 0,
                "largest": max(sizes) if sizes else 0,
                "gt_10": sum(1 for s in sizes if s > 10),
                "gt_50": sum(1 for s in sizes if s > 50),
                "gt_100": sum(1 for s in sizes if s > 100),
            }

        # Print comparison table
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Res':<6} {'Comms':<8} {'Coverage':<10} {'Largest':<8} {'>10':<6} {'>50':<6} {'>100':<6}")
        print("-" * 70)
        for res in resolutions:
            r = results[res]
            print(f"{res:<6.1f} {r['num_communities']:<8} {r['coverage_pct']:<9.1f}% {r['largest']:<8} {r['gt_10']:<6} {r['gt_50']:<6} {r['gt_100']:<6}")

        return results

    def build_and_store_communities(
        self,
        resolution: float = 1.0,
        summarize: bool = True,
        min_community_size_for_summary: int = 5,
        min_edge_weight: int = 1,
        max_node_degree_pct: float = 0.01,
        use_weight_transform: bool = True,
    ):
        """
        Full pipeline: detect communities, summarize, embed, store in DB.

        Args:
            resolution: Leiden resolution parameter
            summarize: Whether to generate LLM summaries
            min_community_size_for_summary: Only summarize communities >= this size (but store ALL)
            min_edge_weight: Minimum edge weight for clustering
            max_node_degree_pct: Max degree percentage for stop entity filtering
        """
        self.detect(
            resolution=resolution,
            min_edge_weight=min_edge_weight,
            max_node_degree_pct=max_node_degree_pct,
            use_weight_transform=use_weight_transform,
        )
        self.storage.ensure_communities_table()

        # Store ALL communities (for membership coverage)
        all_communities = list(self.communities.items())
        communities_to_summarize = [
            (cid, nodes) for cid, nodes in all_communities
            if len(nodes) >= min_community_size_for_summary
        ]

        print(f"\n📝 Storing {len(all_communities)} total communities...")
        print(f"   Summarizing {len(communities_to_summarize)} communities (size >= {min_community_size_for_summary})")

        summarized_count = 0
        for i, (cid, node_ids) in enumerate(all_communities):
            top_terms = []
            summary = None
            summary_embedding = None

            # Only compute expensive operations for larger communities
            if len(node_ids) >= min_community_size_for_summary:
                top_terms = self.get_community_top_terms(cid, top_k=10)

                if summarize and self.llm_client:
                    try:
                        summary = self.summarize_community(cid)
                        summary_embedding = self.get_embedding(summary)
                        summarized_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Failed to summarize community {cid}: {e}")

            self.storage.insert_community(
                community_id=cid,
                level=0,  # Single-level for now
                node_ids=node_ids,
                size=len(node_ids),
                summary=summary,
                summary_embedding=summary_embedding,
                top_terms=top_terms,
            )

            if (i + 1) % 100 == 0:
                print(f"   Stored {i + 1}/{len(all_communities)} communities")

        print(f"✅ Stored {len(all_communities)} communities ({summarized_count} summarized)")

    def get_node_community(self, node_id: str) -> int | None:
        """Get community ID for a node."""
        return self.node_to_community.get(node_id)

    def get_community_nodes(self, community_id: int) -> list[str]:
        """Get all node IDs in a community."""
        return self.communities.get(community_id, [])
