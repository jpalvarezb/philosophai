"""Build NetworkX graph from DuckDB triples."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from ..storage import DuckDBStorage


class GraphBuilder:
    """Builds and manages the knowledge graph."""

    def __init__(self, storage: DuckDBStorage):
        self.storage = storage
        self.graph: nx.MultiDiGraph | None = None

    def build(self) -> nx.MultiDiGraph:
        """
        Build full directed graph from triples in DuckDB.
        Used for traversal and reasoning - preserves all predicates.
        """
        print("📊 Loading triples from DB...")
        df = self.storage.get_triples_df()
        print(f"✅ Loaded {len(df)} unique edges.")

        print("🏗️ Building NetworkX graph...")
        G = nx.MultiDiGraph()

        for _, row in df.iterrows():
            subj_raw = row["subject_canon_id"]
            pred_raw = row["predicate_canon_id"]
            obj_raw = row["object_canon_id"]

            if subj_raw is None or obj_raw is None or pred_raw is None:
                continue

            subj_id = str(subj_raw)
            pred_id = str(pred_raw)
            obj_id = str(obj_raw)

            if not subj_id or not obj_id or not pred_id:
                continue

            subj_label = (
                row.get("subject_label")
                if hasattr(row, "get")
                else row["subject_label"]
            )
            obj_label = (
                row.get("object_label") if hasattr(row, "get") else row["object_label"]
            )
            pred_label = (
                row.get("predicate_label")
                if hasattr(row, "get")
                else row["predicate_label"]
            )

            # Add nodes with labels
            if subj_id not in G:
                G.add_node(subj_id, label=str(subj_label) if subj_label else subj_id)
            if obj_id not in G:
                G.add_node(obj_id, label=str(obj_label) if obj_label else obj_id)

            # Add edge
            G.add_edge(
                subj_id,
                obj_id,
                key=pred_id,
                label=str(pred_label) if pred_label else pred_id,
                weight=int(row["weight"]) if row.get("weight") is not None else 1,
                chunks=(
                    list(row["chunk_ids"]) if row.get("chunk_ids") is not None else []
                ),
            )

        print(
            f"🎉 Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        self.graph = G
        return G

    def build_cluster_projection(self) -> nx.Graph:
        """
        Build undirected graph for clustering.

        Collapses all predicates between entity pairs into single edges.
        Weight = number of distinct chunks mentioning any relationship.

        Use this for Leiden community detection, NOT for traversal.
        """
        print("📊 Loading cluster projection edges...")
        df = self.storage.get_cluster_edges_df()
        print(f"✅ Loaded {len(df)} entity pair edges.")

        # Show weight distribution
        weight_counts = df["weight"].value_counts().sort_index()
        print("   Weight distribution:")
        for w in sorted(weight_counts.index)[:8]:
            print(f"      weight={w}: {weight_counts[w]:,} edges")
        total = len(df)
        gt1 = len(df[df["weight"] > 1])
        print(f"   Edges with weight > 1: {gt1:,} ({100*gt1/total:.1f}%)")

        print("🏗️ Building undirected cluster graph...")
        G = nx.Graph()  # Undirected for clustering

        # Get node labels from the full triples
        node_labels = {}
        triples_df = self.storage.get_triples_df()
        for _, row in triples_df.iterrows():
            node_labels[row["subject_canon_id"]] = row["subject_label"]
            node_labels[row["object_canon_id"]] = row["object_label"]

        for _, row in df.iterrows():
            u_raw = row["u_id"]
            v_raw = row["v_id"]
            if u_raw is None or v_raw is None:
                continue
            u_id = str(u_raw)
            v_id = str(v_raw)
            if not u_id or not v_id:
                continue

            # Add nodes
            if u_id not in G:
                G.add_node(u_id, label=str(node_labels.get(u_id, u_id)))
            if v_id not in G:
                G.add_node(v_id, label=str(node_labels.get(v_id, v_id)))

            # Add undirected edge with aggregated weight
            G.add_edge(
                u_id,
                v_id,
                weight=int(row["weight"]) if row.get("weight") is not None else 1,
                predicate_count=(
                    int(row["predicate_count"])
                    if row.get("predicate_count") is not None
                    else 0
                ),
                top_predicates=(
                    list(row["top_predicates"])
                    if row.get("top_predicates") is not None
                    else []
                ),
                chunks=(
                    list(row["chunk_ids"]) if row.get("chunk_ids") is not None else []
                ),
            )

        print(
            f"🎉 Cluster graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    def save(self, path: str | Path):
        """Save graph to pickle file."""
        if self.graph is None:
            raise ValueError("No graph to save. Call build() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"💾 Graph saved to {path}")

    def load(self, path: str | Path) -> nx.MultiDiGraph:
        """Load graph from pickle file."""
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes")
        return self.graph

    def save_graphml(self, path: str | Path):
        """Save graph to GraphML (for visualization tools)."""
        if self.graph is None:
            raise ValueError("No graph to save.")
        G_export = self.graph.copy()
        # Flatten chunk lists for XML compatibility
        for u, v, k, d in G_export.edges(keys=True, data=True):
            if "chunks" in d and isinstance(d["chunks"], list):
                d["chunks"] = ",".join(d["chunks"])
        nx.write_graphml(G_export, path)
        print(f"💾 GraphML saved to {path}")

    def get_node_label(self, node_id: str) -> str:
        """Get human-readable label for a node."""
        if self.graph is None:
            return node_id
        return self.graph.nodes.get(node_id, {}).get("label", node_id)

    def get_neighbors(self, node_id: str) -> list[tuple[str, str, dict]]:
        """Get outbound neighbors with edge data. Returns (neighbor_id, predicate, edge_attrs)."""
        if self.graph is None or node_id not in self.graph:
            return []
        result = []
        for neighbor in self.graph[node_id]:
            for pred_key, attrs in self.graph[node_id][neighbor].items():
                result.append((neighbor, pred_key, attrs))
        return result

    def get_edge_chunks(
        self, from_id: str, to_id: str, predicate: str | None = None
    ) -> list[str]:
        """Get chunk IDs associated with an edge."""
        if self.graph is None or from_id not in self.graph:
            return []
        if to_id not in self.graph[from_id]:
            return []
        edges = self.graph[from_id][to_id]
        if predicate and predicate in edges:
            return edges[predicate].get("chunks", [])
        # Return all chunks from all predicates between these nodes
        all_chunks = []
        for attrs in edges.values():
            all_chunks.extend(attrs.get("chunks", []))
        return all_chunks
