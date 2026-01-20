"""Community reports and statistics."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from ..storage import DuckDBStorage


class CommunityReporter:
    """Generate reports and statistics about communities."""

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
            if not top_terms:
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
            terms = row["top_terms"][:5] if row["top_terms"] else []
            print(f"  Terms: {', '.join(terms)}")
            if row["summary"]:
                summary = row["summary"][:200] + "..." if len(row["summary"]) > 200 else row["summary"]
                print(f"  Summary: {summary}")
