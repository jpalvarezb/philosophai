"""DuckDB storage layer for PhilosophAI."""
from __future__ import annotations

import duckdb
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class DuckDBStorage:
    """Manages DuckDB connection and provides query methods."""

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._con: duckdb.DuckDBPyConnection | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self._con

    def close(self):
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # Chunks
    # -------------------------------------------------------------------------
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> pd.DataFrame:
        """Fetch chunks by their IDs."""
        if not chunk_ids:
            return self.con.execute("SELECT * FROM chunks WHERE 1=0").fetchdf()
        placeholders = ",".join(["?"] * len(chunk_ids))
        return self.con.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchdf()

    def get_chunk_texts(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Return list of (chunk_id, content) tuples."""
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        return self.con.execute(
            f"SELECT chunk_id, content FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

    # -------------------------------------------------------------------------
    # Vector search
    # -------------------------------------------------------------------------
    def vector_search_chunks(
        self,
        query_embedding: list[float],
        limit: int = 10,
        text_ids: set[int] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find top-k chunks by cosine similarity.
        
        Args:
            query_embedding: Query vector
            limit: Max results to return
            text_ids: If provided, restrict to chunks from these text_ids (scope filter)
        
        Returns:
            List of (chunk_id, score) tuples
        """
        if text_ids is not None and len(text_ids) > 0:
            # Scoped search: join with chunks table to filter by text_id
            placeholders = ",".join(["?"] * len(text_ids))
            sql = f"""
                SELECT ec.chunk_id, list_cosine_similarity(ec.embedding, ?::DOUBLE[]) as score
                FROM embedded_chunks ec
                INNER JOIN chunks c ON ec.chunk_id = c.chunk_id
                WHERE c.text_id IN ({placeholders})
                ORDER BY score DESC
                LIMIT ?
            """
            return self.con.execute(sql, [query_embedding] + list(text_ids) + [limit]).fetchall()
        else:
            # Unscoped search
            sql = """
                SELECT chunk_id, list_cosine_similarity(embedding, ?::DOUBLE[]) as score
                FROM embedded_chunks
                ORDER BY score DESC
                LIMIT ?
            """
            return self.con.execute(sql, [query_embedding, limit]).fetchall()

    # -------------------------------------------------------------------------
    # Triples / Graph seeds
    # -------------------------------------------------------------------------
    def get_triples_df(self) -> pd.DataFrame:
        """Fetch all cleaned/canonicalized triples for graph building."""
        return self.con.execute("""
            SELECT
                subject_canon_id,
                predicate_canon_id,
                object_canon_id,
                mode(subject_norm) as subject_label,
                mode(predicate_norm) as predicate_label,
                mode(object_norm) as object_label,
                COUNT(DISTINCT chunk_id) as weight,
                list(DISTINCT chunk_id) as chunk_ids
            FROM normalized_triples_clean_canon
            WHERE object_norm IS NOT NULL AND object_norm != ''
            GROUP BY 1, 2, 3
        """).fetchdf()

    def get_entity_ids_from_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Find all entity IDs (subject/object) mentioned in given chunks."""
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT DISTINCT subject_canon_id FROM normalized_triples_clean_canon 
            WHERE chunk_id IN ({placeholders})
            UNION
            SELECT DISTINCT object_canon_id FROM normalized_triples_clean_canon 
            WHERE chunk_id IN ({placeholders})
        """
        results = self.con.execute(sql, chunk_ids * 2).fetchall()
        return [r[0] for r in results if r[0]]

    def get_cluster_edges_df(self) -> "pd.DataFrame":
        """
        Get edges aggregated by (subject, object) pair for clustering.
        
        Collapses all predicates between an entity pair into a single undirected edge.
        Weight = number of distinct chunks mentioning any relationship between the pair.
        
        Returns DataFrame with columns:
            - u_id: lesser of the two entity IDs (for undirected consistency)
            - v_id: greater of the two entity IDs
            - weight: COUNT(DISTINCT chunk_id) across all predicates
            - predicate_count: number of distinct predicates
            - top_predicates: most common predicates (for reporting)
            - chunk_ids: list of supporting chunk IDs
        """
        return self.con.execute("""
            SELECT
                LEAST(subject_canon_id, object_canon_id) as u_id,
                GREATEST(subject_canon_id, object_canon_id) as v_id,
                COUNT(DISTINCT chunk_id) as weight,
                COUNT(DISTINCT predicate_canon_id) as predicate_count,
                ARRAY_AGG(DISTINCT predicate_canon_id ORDER BY predicate_canon_id)[:5] as top_predicates,
                LIST(DISTINCT chunk_id) as chunk_ids
            FROM normalized_triples_clean_canon
            WHERE object_canon_id IS NOT NULL AND object_canon_id != ''
              AND subject_canon_id != object_canon_id  -- no self-loops
            GROUP BY 1, 2
        """).fetchdf()

    def get_pair_predicates(self, u_id: str, v_id: str) -> list[tuple[str, int]]:
        """
        Get all predicates between an entity pair with their support counts.
        Used for community reports to show predicate distribution.
        
        Returns list of (predicate, count) tuples sorted by count desc.
        """
        result = self.con.execute("""
            SELECT predicate_canon_id, COUNT(DISTINCT chunk_id) as support
            FROM normalized_triples_clean_canon
            WHERE (
                (subject_canon_id = ? AND object_canon_id = ?)
                OR (subject_canon_id = ? AND object_canon_id = ?)
            )
            GROUP BY 1
            ORDER BY support DESC
        """, [u_id, v_id, v_id, u_id]).fetchall()
        return result

    # -------------------------------------------------------------------------
    # Scoped Graph Queries (for author/tradition/domain filtering)
    # -------------------------------------------------------------------------
    def get_scoped_edges(
        self, chunk_ids: set[str]
    ) -> set[tuple[str, str, str]]:
        """
        Get all edges that have provenance in the given chunks.
        
        An edge (subject, object, predicate) is "in scope" if at least one
        of its supporting chunks is in the scoped set.
        
        Args:
            chunk_ids: Set of chunk IDs defining the scope
        
        Returns:
            Set of (subject_canon_id, object_canon_id, predicate_canon_id) tuples
        """
        if not chunk_ids:
            return set()
        
        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT DISTINCT 
                subject_canon_id, 
                object_canon_id, 
                predicate_canon_id
            FROM normalized_triples_clean_canon
            WHERE chunk_id IN ({placeholders})
              AND object_canon_id IS NOT NULL 
              AND object_canon_id != ''
        """
        results = self.con.execute(sql, list(chunk_ids)).fetchall()
        return {(r[0], r[1], r[2]) for r in results}

    def _get_scoped_entity_ids(self, chunk_ids: set[str]) -> set[str]:
        """
        Internal: Get all entity IDs that appear in edges supported by scoped chunks.
        
        A node is "in scope" if it participates in at least one in-scope edge.
        This is used internally by derive_scoped_communities.
        
        Note: For traversal, entity scope should be derived from scoped_edges as V(E),
        not queried separately.
        
        Args:
            chunk_ids: Set of chunk IDs defining the scope
        
        Returns:
            Set of entity IDs (both subjects and objects)
        """
        if not chunk_ids:
            return set()
        
        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT DISTINCT subject_canon_id FROM normalized_triples_clean_canon 
            WHERE chunk_id IN ({placeholders})
              AND object_canon_id IS NOT NULL AND object_canon_id != ''
            UNION
            SELECT DISTINCT object_canon_id FROM normalized_triples_clean_canon 
            WHERE chunk_id IN ({placeholders})
              AND object_canon_id IS NOT NULL AND object_canon_id != ''
        """
        results = self.con.execute(sql, list(chunk_ids) * 2).fetchall()
        return {r[0] for r in results if r[0]}

    def derive_scoped_communities(
        self,
        chunk_ids: set[str],
        node_to_community: dict[str, int],
        top_n: int = 5,
    ) -> list[tuple[int, int]]:
        """
        Derive relevant communities from scoped chunks.
        
        Maps: scoped chunks -> entities in those chunks -> their global communities
        -> ranked by overlap count.
        
        This replaces global community report routing for scoped queries.
        
        Args:
            chunk_ids: Set of chunk IDs defining the scope
            node_to_community: Global node->community mapping
            top_n: Number of top communities to return
        
        Returns:
            List of (community_id, entity_count) tuples, sorted by count desc
        """
        if not chunk_ids:
            return []
        
        # Get entities from scoped chunks (internal helper)
        entity_ids = self._get_scoped_entity_ids(chunk_ids)
        
        # Count entities per community
        from collections import Counter
        community_counts: Counter[int] = Counter()
        for entity_id in entity_ids:
            comm_id = node_to_community.get(entity_id)
            if comm_id is not None:
                community_counts[comm_id] += 1
        
        # Return top N communities by entity count
        return community_counts.most_common(top_n)

    # -------------------------------------------------------------------------
    # Communities (to be populated later)
    # -------------------------------------------------------------------------
    def ensure_communities_table(self):
        """Create communities table if not exists."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS communities (
                community_id INTEGER PRIMARY KEY,
                level INTEGER,
                node_ids VARCHAR[],
                size INTEGER,
                summary TEXT,
                summary_embedding DOUBLE[],
                top_terms VARCHAR[]
            )
        """)

    def insert_community(
        self,
        community_id: int,
        level: int,
        node_ids: list[str],
        size: int,
        summary: str | None = None,
        summary_embedding: list[float] | None = None,
        top_terms: list[str] | None = None,
    ):
        """Insert or replace a community record."""
        self.con.execute(
            """
            INSERT OR REPLACE INTO communities 
            (community_id, level, node_ids, size, summary, summary_embedding, top_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [community_id, level, node_ids, size, summary, summary_embedding, top_terms],
        )

    def get_communities(self) -> pd.DataFrame:
        """Fetch all communities."""
        return self.con.execute("SELECT * FROM communities").fetchdf()

    def vector_search_communities(
        self, query_embedding: list[float], limit: int = 5
    ) -> list[tuple[int, float]]:
        """Find top-k communities by summary embedding similarity."""
        sql = """
            SELECT community_id, list_cosine_similarity(summary_embedding, ?::DOUBLE[]) as score
            FROM communities
            WHERE summary_embedding IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """
        return self.con.execute(sql, [query_embedding, limit]).fetchall()

    # -------------------------------------------------------------------------
    # Community Membership (normalized for efficient lookups)
    # -------------------------------------------------------------------------
    def ensure_community_membership_table(self):
        """Create community_membership table if not exists."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS community_membership (
                comm_id INTEGER,
                node_id VARCHAR,
                weight DOUBLE DEFAULT 1.0,
                PRIMARY KEY (comm_id, node_id)
            )
        """)
        # Index for node lookups
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_membership_node 
            ON community_membership(node_id)
        """)

    def populate_community_membership(self):
        """
        Populate community_membership from communities.node_ids arrays.
        Call after running Leiden detection.
        """
        self.ensure_community_membership_table()
        # Clear existing
        self.con.execute("DELETE FROM community_membership")
        # Unnest node_ids arrays into normalized rows
        self.con.execute("""
            INSERT INTO community_membership (comm_id, node_id, weight)
            SELECT community_id, UNNEST(node_ids), 1.0
            FROM communities
        """)
        count = self.con.execute("SELECT COUNT(*) FROM community_membership").fetchone()[0]
        print(f"✅ Populated {count} community membership records")

    def get_nodes_in_communities(self, comm_ids: list[int]) -> list[str]:
        """Get all node IDs belonging to the given communities."""
        if not comm_ids:
            return []
        placeholders = ",".join(["?"] * len(comm_ids))
        sql = f"""
            SELECT DISTINCT node_id 
            FROM community_membership 
            WHERE comm_id IN ({placeholders})
        """
        results = self.con.execute(sql, comm_ids).fetchall()
        return [r[0] for r in results]

    def get_chunks_for_communities(self, comm_ids: list[int]) -> list[str]:
        """
        Get all chunk IDs associated with entities in the given communities.
        Uses entity->chunk mapping from triples.
        """
        if not comm_ids:
            return []
        placeholders = ",".join(["?"] * len(comm_ids))
        sql = f"""
            SELECT DISTINCT t.chunk_id
            FROM normalized_triples_clean_canon t
            INNER JOIN community_membership m 
                ON t.subject_canon_id = m.node_id OR t.object_canon_id = m.node_id
            WHERE m.comm_id IN ({placeholders})
        """
        results = self.con.execute(sql, comm_ids).fetchall()
        return [r[0] for r in results if r[0]]

    def get_community_for_node(self, node_id: str) -> int | None:
        """Get community ID for a single node."""
        result = self.con.execute(
            "SELECT comm_id FROM community_membership WHERE node_id = ? LIMIT 1",
            [node_id],
        ).fetchone()
        return result[0] if result else None

    # -------------------------------------------------------------------------
    # Community Reports (Phase 3.2)
    # -------------------------------------------------------------------------
    def ensure_community_reports_table(self):
        """Create community_reports table if not exists."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS community_reports (
                comm_id INTEGER PRIMARY KEY,
                report_text TEXT,
                report_embedding DOUBLE[],
                cited_chunk_ids VARCHAR[],
                entity_ids VARCHAR[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def insert_community_report(
        self,
        comm_id: int,
        report_text: str,
        report_embedding: list[float] | None = None,
        cited_chunk_ids: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ):
        """Insert or replace a community report."""
        self.con.execute(
            """
            INSERT OR REPLACE INTO community_reports 
            (comm_id, report_text, report_embedding, cited_chunk_ids, entity_ids, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [comm_id, report_text, report_embedding, cited_chunk_ids or [], entity_ids or []],
        )

    def get_community_reports(self) -> pd.DataFrame:
        """Fetch all community reports."""
        try:
            return self.con.execute("SELECT * FROM community_reports").fetchdf()
        except Exception:
            # Table doesn't exist yet
            import pandas as pd
            return pd.DataFrame()

    def vector_search_community_reports(
        self,
        query_embedding: list[float],
        limit: int = 5,
        text_ids: set[int] | None = None,
    ) -> list[tuple[int, float, list[str]]]:
        """
        Find top-k community reports by embedding similarity.
        
        Args:
            query_embedding: Query vector
            limit: Max results to return
            text_ids: If provided, filter cited_chunk_ids to only those from these texts
        
        Returns:
            List of (comm_id, score, cited_chunk_ids) tuples.
            When text_ids is provided, cited_chunk_ids are pre-filtered at SQL level.
        """
        try:
            if text_ids is not None and len(text_ids) > 0:
                # Scoped search: filter cited_chunk_ids at SQL level
                text_placeholders = ",".join(["?"] * len(text_ids))
                sql = f"""
                    WITH ranked_reports AS (
                        SELECT 
                            comm_id, 
                            list_cosine_similarity(report_embedding, ?::DOUBLE[]) as score,
                            cited_chunk_ids
                        FROM community_reports
                        WHERE report_embedding IS NOT NULL
                        ORDER BY score DESC
                        LIMIT ?
                    ),
                    in_scope_chunks AS (
                        SELECT chunk_id FROM chunks WHERE text_id IN ({text_placeholders})
                    )
                    SELECT 
                        r.comm_id,
                        r.score,
                        ARRAY_AGG(uc.chunk_id) FILTER (WHERE uc.chunk_id IS NOT NULL) as cited_chunk_ids
                    FROM ranked_reports r
                    LEFT JOIN LATERAL UNNEST(r.cited_chunk_ids) AS uc(chunk_id) ON TRUE
                    LEFT JOIN in_scope_chunks isc ON uc.chunk_id = isc.chunk_id
                    WHERE uc.chunk_id IS NULL OR isc.chunk_id IS NOT NULL
                    GROUP BY r.comm_id, r.score
                    ORDER BY r.score DESC
                """
                params = [query_embedding, limit] + list(text_ids)
                results = self.con.execute(sql, params).fetchall()
                # Convert None to empty list for cited_chunk_ids
                return [(r[0], r[1], r[2] if r[2] else []) for r in results]
            else:
                # Unscoped search
                sql = """
                    SELECT 
                        comm_id, 
                        list_cosine_similarity(report_embedding, ?::DOUBLE[]) as score,
                        cited_chunk_ids
                    FROM community_reports
                    WHERE report_embedding IS NOT NULL
                    ORDER BY score DESC
                    LIMIT ?
                """
                return self.con.execute(sql, [query_embedding, limit]).fetchall()
        except Exception:
            # Table doesn't exist yet
            return []
