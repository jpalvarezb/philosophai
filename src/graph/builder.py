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
        """Build graph from triples in DuckDB."""
        print("📊 Loading triples from DB...")
        df = self.storage.get_triples_df()
        print(f"✅ Loaded {len(df)} unique edges.")

        print("🏗️ Building NetworkX graph...")
        G = nx.MultiDiGraph()

        for _, row in df.iterrows():
            subj_id = row["subject_canon_id"]
            pred_id = row["predicate_canon_id"]
            obj_id = row["object_canon_id"]

            # Add nodes with labels
            if subj_id not in G:
                G.add_node(subj_id, label=row["subject_label"])
            if obj_id not in G:
                G.add_node(obj_id, label=row["object_label"])

            # Add edge
            G.add_edge(
                subj_id,
                obj_id,
                key=pred_id,
                label=row["predicate_label"],
                weight=row["weight"],
                chunks=row["chunk_ids"],
            )

        print(f"🎉 Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        self.graph = G
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

    def get_edge_chunks(self, from_id: str, to_id: str, predicate: str | None = None) -> list[str]:
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
