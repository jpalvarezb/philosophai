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
    """Detects communities in the knowledge graph using Leiden algorithm."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        storage: "DuckDBStorage",
        llm_client: "OpenAI | None" = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.graph = graph
        self.storage = storage
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.communities: dict[int, list[str]] = {}  # community_id -> node_ids
        self.node_to_community: dict[str, int] = {}  # node_id -> community_id

    def detect(self, resolution: float = 1.0) -> dict[int, list[str]]:
        """
        Run Leiden community detection.
        
        Args:
            resolution: Higher values = more/smaller communities. Default 1.0.
        
        Returns:
            Dict mapping community_id -> list of node_ids
        """
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "leidenalg and python-igraph required. "
                "Install with: pip install leidenalg python-igraph"
            )

        print("🔍 Converting NetworkX to igraph...")
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        # Create igraph from NetworkX
        node_list = list(G_undirected.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        edges = [
            (node_to_idx[u], node_to_idx[v])
            for u, v in G_undirected.edges()
        ]
        
        ig_graph = ig.Graph(n=len(node_list), edges=edges)
        ig_graph.vs["name"] = node_list

        print(f"🧮 Running Leiden algorithm (resolution={resolution})...")
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )

        # Build community mappings
        self.communities = defaultdict(list)
        for idx, community_id in enumerate(partition.membership):
            node_id = node_list[idx]
            self.communities[community_id].append(node_id)
            self.node_to_community[node_id] = community_id

        print(f"✅ Found {len(self.communities)} communities")
        
        # Print size distribution
        sizes = sorted([len(nodes) for nodes in self.communities.values()], reverse=True)
        print(f"   Largest: {sizes[0]} nodes")
        print(f"   Top 5 sizes: {sizes[:5]}")
        print(f"   Communities with >10 nodes: {sum(1 for s in sizes if s > 10)}")

        return dict(self.communities)

    def get_community_top_terms(self, community_id: int, top_k: int = 10) -> list[str]:
        """Get most representative node labels for a community."""
        if community_id not in self.communities:
            return []
        
        node_ids = self.communities[community_id]
        # Score by degree within community
        subgraph = self.graph.subgraph(node_ids)
        degree_scores = dict(subgraph.degree())
        
        # Sort by degree, get labels
        sorted_nodes = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n[0] for n in sorted_nodes[:top_k]]
        
        labels = []
        for node_id in top_nodes:
            label = self.graph.nodes.get(node_id, {}).get("label", node_id)
            labels.append(label)
        return labels

    def summarize_community(self, community_id: int) -> str:
        """Generate LLM summary for a community based on its top terms and structure."""
        if self.llm_client is None:
            raise ValueError("LLM client required for summarization")
        
        top_terms = self.get_community_top_terms(community_id, top_k=15)
        node_ids = self.communities.get(community_id, [])
        
        # Get sample triples from this community
        sample_triples = []
        subgraph = self.graph.subgraph(node_ids)
        for u, v, data in list(subgraph.edges(data=True))[:20]:
            u_label = self.graph.nodes[u].get("label", u)
            v_label = self.graph.nodes[v].get("label", v)
            pred_label = data.get("label", "relates to")
            sample_triples.append(f"({u_label}, {pred_label}, {v_label})")

        prompt = f"""Summarize this knowledge graph community in 2-3 sentences.

Community contains {len(node_ids)} entities. 

Key terms: {', '.join(top_terms)}

Sample relationships:
{chr(10).join(sample_triples[:15])}

Write a concise summary describing what this community is about - the main themes, concepts, or domain it covers."""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
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

    def build_and_store_communities(
        self,
        resolution: float = 1.0,
        summarize: bool = True,
        min_community_size: int = 5,
    ):
        """
        Full pipeline: detect communities, summarize, embed, store in DB.
        
        Args:
            resolution: Leiden resolution parameter
            summarize: Whether to generate LLM summaries
            min_community_size: Only process communities with at least this many nodes
        """
        self.detect(resolution=resolution)
        self.storage.ensure_communities_table()

        communities_to_process = [
            (cid, nodes)
            for cid, nodes in self.communities.items()
            if len(nodes) >= min_community_size
        ]
        
        print(f"\n📝 Processing {len(communities_to_process)} communities (size >= {min_community_size})...")

        for i, (cid, node_ids) in enumerate(communities_to_process):
            top_terms = self.get_community_top_terms(cid, top_k=10)
            
            summary = None
            summary_embedding = None
            
            if summarize and self.llm_client:
                try:
                    summary = self.summarize_community(cid)
                    summary_embedding = self.get_embedding(summary)
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

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(communities_to_process)} communities")

        print(f"✅ Stored {len(communities_to_process)} communities in DB")

    def get_node_community(self, node_id: str) -> int | None:
        """Get community ID for a node."""
        return self.node_to_community.get(node_id)

    def get_community_nodes(self, community_id: int) -> list[str]:
        """Get all node IDs in a community."""
        return self.communities.get(community_id, [])
