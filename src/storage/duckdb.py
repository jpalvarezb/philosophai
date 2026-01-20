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
        self, query_embedding: list[float], limit: int = 10
    ) -> list[tuple[str, float]]:
        """Find top-k chunks by cosine similarity. Returns (chunk_id, score)."""
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
