"""Community reports generation with chunk citations for GraphRAG routing."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import networkx as nx
    from openai import OpenAI
    from ..storage import DuckDBStorage


class CommunityReportGenerator:
    """
    Generate grounded community reports with chunk citations.

    Each report contains:
    - Representative entities (by weighted degree)
    - Key relationships (top edges by support)
    - Themes/topics inferred from entity labels
    - Cited chunk_ids for grounding
    """

    def __init__(
        self,
        storage: "DuckDBStorage",
        graph: "nx.MultiDiGraph",
        node_to_community: dict[str, int],
        llm_client: "OpenAI",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.storage = storage
        self.graph = graph
        self.node_to_community = node_to_community
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    def generate_all_reports(
        self,
        min_community_size: int = 5,
        max_communities: int | None = None,
    ):
        """
        Generate reports for all communities meeting size threshold.

        Args:
            min_community_size: Skip communities smaller than this
            max_communities: Limit number of communities to process (for testing)
        """
        self.storage.ensure_community_reports_table()

        communities_df = self.storage.get_communities()
        if communities_df.empty:
            print("⚠️ No communities found. Run Leiden detection first.")
            return

        # Filter by size
        eligible = communities_df[communities_df["size"] >= min_community_size]
        if max_communities:
            eligible = eligible.nlargest(max_communities, "size")

        print(f"📝 Generating reports for {len(eligible)} communities...")

        for i, (_, row) in enumerate(eligible.iterrows()):
            comm_id = row["community_id"]
            node_ids = row["node_ids"]

            try:
                report = self._generate_single_report(comm_id, node_ids)

                # Embed the report
                embedding = self._get_embedding(report["report_text"])

                # Store
                self.storage.insert_community_report(
                    comm_id=comm_id,
                    report_text=report["report_text"],
                    report_embedding=embedding,
                    cited_chunk_ids=report["cited_chunk_ids"],
                    entity_ids=report["entity_ids"],
                )

                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(eligible)} communities")

            except Exception as e:
                print(f"   ⚠️ Failed to generate report for community {comm_id}: {e}")

        print(f"✅ Generated {len(eligible)} community reports")

    def _generate_single_report(self, comm_id: int, node_ids: list[str]) -> dict:
        """
        Generate a report for a single community.

        Returns:
            {
                "report_text": str,
                "cited_chunk_ids": list[str],
                "entity_ids": list[str],
            }
        """
        # 1. Get representative entities by weighted degree
        top_entities = self._get_top_entities(node_ids, top_k=15)
        entity_ids = [e["id"] for e in top_entities]

        # 2. Get top edges with chunk provenance
        top_edges = self._get_top_edges(node_ids, top_k=20)

        # 3. Collect cited chunk_ids from edges
        cited_chunk_ids = []
        for edge in top_edges:
            cited_chunk_ids.extend(edge["chunk_ids"])
        cited_chunk_ids = list(set(cited_chunk_ids))[:50]  # Cap at 50

        # 4. Generate report text via LLM
        report_text = self._generate_report_text(
            comm_id=comm_id,
            entities=top_entities,
            edges=top_edges,
            total_nodes=len(node_ids),
        )

        return {
            "report_text": report_text,
            "cited_chunk_ids": cited_chunk_ids,
            "entity_ids": entity_ids,
        }

    def _get_top_entities(self, node_ids: list[str], top_k: int = 15) -> list[dict]:
        """Get top entities by weighted degree within community."""
        node_set = set(node_ids)
        subgraph = self.graph.subgraph(node_ids)

        # Calculate weighted degree (sum of edge weights)
        weighted_degrees = {}
        for node in subgraph.nodes():
            total_weight = 0
            # Outbound
            for neighbor in subgraph.successors(node):
                for key, data in subgraph[node][neighbor].items():
                    total_weight += data.get("weight", 1)
            # Inbound
            for neighbor in subgraph.predecessors(node):
                for key, data in subgraph[neighbor][node].items():
                    total_weight += data.get("weight", 1)
            weighted_degrees[node] = total_weight

        # Sort and take top k
        sorted_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)

        result = []
        for node_id, degree in sorted_nodes[:top_k]:
            label = self.graph.nodes.get(node_id, {}).get("label", node_id)
            result.append({
                "id": node_id,
                "label": label,
                "weighted_degree": degree,
            })
        return result

    def _get_top_edges(self, node_ids: list[str], top_k: int = 20) -> list[dict]:
        """Get top edges by weight/support within community."""
        node_set = set(node_ids)
        edges = []

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            for neighbor in self.graph[node_id]:
                if neighbor not in node_set:
                    continue
                for pred_key, data in self.graph[node_id][neighbor].items():
                    edges.append({
                        "subject_id": node_id,
                        "subject_label": self.graph.nodes.get(node_id, {}).get("label", node_id),
                        "predicate": data.get("label", pred_key),
                        "object_id": neighbor,
                        "object_label": self.graph.nodes.get(neighbor, {}).get("label", neighbor),
                        "weight": data.get("weight", 1),
                        "chunk_ids": data.get("chunks", []),
                    })

        # Sort by weight and take top k
        edges.sort(key=lambda x: x["weight"], reverse=True)
        return edges[:top_k]

    def _generate_report_text(
        self,
        comm_id: int,
        entities: list[dict],
        edges: list[dict],
        total_nodes: int,
    ) -> str:
        """Generate structured report text via LLM."""
        entity_labels = [e["label"] for e in entities[:10]]

        edge_descriptions = []
        for e in edges[:15]:
            edge_descriptions.append(
                f"- {e['subject_label']} {e['predicate']} {e['object_label']} (support: {e['weight']})"
            )

        prompt = f"""Generate a concise knowledge summary for a community cluster in a knowledge graph.

Community ID: {comm_id}
Total entities: {total_nodes}

Key entities (by importance):
{', '.join(entity_labels)}

Key relationships:
{chr(10).join(edge_descriptions)}

Write a 3-5 sentence summary that:
1. Identifies the main theme or domain of this community
2. Highlights the most important concepts and their relationships
3. Notes any notable patterns in how concepts are connected

Be specific and factual based on the entities and relationships shown."""

        response = self.llm_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        text = text.replace("\n", " ")
        response = self.llm_client.embeddings.create(
            input=[text],
            model=self.embedding_model,
        )
        return response.data[0].embedding


class CommunityReporter:
    """Generate reports and statistics about communities (read-only)."""

    def __init__(self, storage: "DuckDBStorage"):
        self.storage = storage

    def get_summary_stats(self) -> dict:
        """Get high-level statistics about communities."""
        df = self.storage.get_communities()
        if df.empty:
            return {"total_communities": 0}

        return {
            "total_communities": len(df),
            "total_nodes": df["size"].sum(),
            "avg_size": df["size"].mean(),
            "median_size": df["size"].median(),
            "max_size": df["size"].max(),
            "min_size": df["size"].min(),
            "with_summaries": df["summary"].notna().sum(),
        }

    def get_largest_communities(self, top_k: int = 10) -> "pd.DataFrame":
        """Get the k largest communities with their summaries."""
        df = self.storage.get_communities()
        if df.empty:
            return df
        return df.nlargest(top_k, "size")[["community_id", "size", "top_terms", "summary"]]

    def find_community_by_terms(self, search_terms: list[str]) -> "pd.DataFrame":
        """Find communities containing specific terms."""
        df = self.storage.get_communities()
        if df.empty:
            return df

        def contains_terms(top_terms):
            if top_terms is None or len(top_terms) == 0:
                return False
            terms_lower = [t.lower() for t in top_terms]
            return any(
                any(search.lower() in term for term in terms_lower)
                for search in search_terms
            )

        mask = df["top_terms"].apply(contains_terms)
        return df[mask][["community_id", "size", "top_terms", "summary"]]

    def get_community_detail(self, community_id: int) -> dict | None:
        """Get full details for a specific community."""
        df = self.storage.get_communities()
        row = df[df["community_id"] == community_id]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            "community_id": row["community_id"],
            "level": row["level"],
            "size": row["size"],
            "node_ids": row["node_ids"],
            "top_terms": row["top_terms"],
            "summary": row["summary"],
            "has_embedding": row["summary_embedding"] is not None,
        }

    def print_report(self):
        """Print a formatted report of community statistics."""
        stats = self.get_summary_stats()

        print("=" * 60)
        print("COMMUNITY REPORT")
        print("=" * 60)

        if stats["total_communities"] == 0:
            print("No communities detected yet.")
            return

        print(f"Total communities: {stats['total_communities']}")
        print(f"Total nodes:       {stats['total_nodes']}")
        print(f"Average size:      {stats['avg_size']:.1f}")
        print(f"Median size:       {stats['median_size']:.1f}")
        print(f"Largest:           {stats['max_size']}")
        print(f"Smallest:          {stats['min_size']}")
        print(f"With summaries:    {stats['with_summaries']}")

        print("\n" + "-" * 60)
        print("TOP 5 LARGEST COMMUNITIES:")
        print("-" * 60)

        largest = self.get_largest_communities(5)
        for _, row in largest.iterrows():
            print(f"\nCommunity {row['community_id']} ({row['size']} nodes)")
            top_terms = row["top_terms"]
            terms = top_terms[:5] if top_terms is not None and len(top_terms) > 0 else []
            print(f"  Terms: {', '.join(terms)}")
            if row["summary"]:
                summary = row["summary"][:200] + "..." if len(row["summary"]) > 200 else row["summary"]
                print(f"  Summary: {summary}")
