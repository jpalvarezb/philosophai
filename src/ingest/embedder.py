"""Chunk embedding generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import IngestConfig

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage


class ChunkEmbedder:
    """Generate and store embeddings for text chunks."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI",
        embedding_model: str = "text-embedding-3-small",
        config: IngestConfig | None = None,
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.config = config or IngestConfig()
        self.embedding_model = embedding_model or self.config.embedding_model

    def embed_chunks(
        self,
        batch_size: int | None = None,
        skip_existing: bool = True,
    ) -> dict:
        """
        Embed all chunks and store in embedded_chunks table.

        Args:
            batch_size: Number of chunks to embed per API call
            skip_existing: If True, only embed chunks not already in embedded_chunks

        Returns:
            Dict with statistics
        """
        con = self.storage.con
        batch_size = batch_size or self.config.embedding_batch_size

        # Ensure table exists
        con.execute("""
            CREATE TABLE IF NOT EXISTS embedded_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                embedding DOUBLE[]
            )
        """)

        # Get chunks to embed
        if skip_existing:
            chunks_df = con.execute("""
                SELECT c.chunk_id, c.content
                FROM chunks c
                LEFT JOIN embedded_chunks e ON c.chunk_id = e.chunk_id
                WHERE e.chunk_id IS NULL
            """).fetchdf()
        else:
            chunks_df = con.execute("SELECT chunk_id, content FROM chunks").fetchdf()

        total = len(chunks_df)
        if total == 0:
            print("✅ All chunks already embedded")
            return {
                "embedded": 0,
                "total_in_table": con.execute(
                    "SELECT COUNT(*) FROM embedded_chunks"
                ).fetchone()[0],
            }

        print(f"📊 Embedding {total:,} chunks...")

        from tqdm import tqdm

        embedded_count = 0

        for i in tqdm(range(0, total, batch_size), desc="Embedding"):
            batch = chunks_df.iloc[i : i + batch_size]
            chunk_ids = batch["chunk_id"].tolist()
            texts = batch["content"].tolist()

            # Get embeddings from OpenAI
            embeddings = self._get_embeddings(texts)

            # Insert into table
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                con.execute(
                    "INSERT OR REPLACE INTO embedded_chunks (chunk_id, embedding) VALUES (?, ?)",
                    [chunk_id, embedding],
                )

            embedded_count += len(chunk_ids)

        total_in_table = con.execute("SELECT COUNT(*) FROM embedded_chunks").fetchone()[
            0
        ]
        print(f"✅ Embedded {embedded_count:,} chunks (total: {total_in_table:,})")

        return {
            "embedded": embedded_count,
            "total_in_table": total_in_table,
        }

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        # Clean texts (remove newlines, truncate if needed)
        cleaned = [t.replace("\n", " ")[:8000] for t in texts]

        response = self.llm_client.embeddings.create(
            input=cleaned,
            model=self.embedding_model,
        )

        return [item.embedding for item in response.data]

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        return self._get_embeddings([text])[0]

    def embed_entities(
        self,
        source_table: str = "entity_canon_map",
        batch_size: int | None = None,
    ) -> dict:
        """
        Embed canonical entities (optional, for entity-level search).

        Creates entity_embeddings table.
        """
        con = self.storage.con
        batch_size = batch_size or self.config.embedding_batch_size

        # Create table
        con.execute("""
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_canon VARCHAR PRIMARY KEY,
                embedding DOUBLE[]
            )
        """)

        # Get entities to embed
        entities_df = con.execute(f"""
            SELECT DISTINCT entity_canon
            FROM {source_table}
            WHERE entity_canon NOT IN (SELECT entity_canon FROM entity_embeddings)
        """).fetchdf()

        total = len(entities_df)
        if total == 0:
            print("✅ All entities already embedded")
            return {"embedded": 0}

        print(f"📊 Embedding {total:,} entities...")

        from tqdm import tqdm

        embedded_count = 0

        for i in tqdm(range(0, total, batch_size), desc="Embedding"):
            batch = entities_df.iloc[i : i + batch_size]
            entities = batch["entity_canon"].tolist()

            # Get embeddings
            embeddings = self._get_embeddings(entities)

            # Insert
            for entity, embedding in zip(entities, embeddings):
                con.execute(
                    "INSERT OR REPLACE INTO entity_embeddings (entity_canon, embedding) VALUES (?, ?)",
                    [entity, embedding],
                )

            embedded_count += len(entities)

        print(f"✅ Embedded {embedded_count:,} entities")
        return {"embedded": embedded_count}
