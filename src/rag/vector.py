"""Vector search for chunks and community summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    chunk_ids: list[str]
    chunk_scores: list[float]
    community_ids: list[int]
    community_scores: list[float]
    query_embedding: list[float]


@dataclass
class CommunityReportSearchResult:
    """Result from community report search (GraphRAG routing)."""
    community_ids: list[int]  # Matched community IDs
    scores: list[float]  # Similarity scores
    cited_chunk_ids: list[str]  # Chunks cited in matched reports
    query_embedding: list[float]


class VectorSearch:
    """Perform vector similarity search on chunks and community summaries."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        text = text.replace("\n", " ")
        response = self.llm_client.embeddings.create(
            input=[text],
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        chunk_limit: int = 10,
        community_limit: int = 5,
        text_ids: set[int] | None = None,
    ) -> VectorSearchResult:
        """
        Search both chunks and community summaries.
        
        Args:
            query: Search query text
            chunk_limit: Max chunks to return
            community_limit: Max communities to return
            text_ids: If provided, restrict chunks to those from these text_ids (scope filter)
        
        Returns:
            VectorSearchResult with ranked chunks and communities
        """
        query_embedding = self.get_embedding(query)

        # Search chunks (scoped if text_ids provided)
        chunk_results = self.storage.vector_search_chunks(
            query_embedding, limit=chunk_limit, text_ids=text_ids
        )
        chunk_ids = [r[0] for r in chunk_results]
        chunk_scores = [r[1] for r in chunk_results]

        # Search communities (unscoped - communities aggregate across texts)
        community_results = self.storage.vector_search_communities(
            query_embedding, limit=community_limit
        )
        community_ids = [r[0] for r in community_results]
        community_scores = [r[1] for r in community_results]

        return VectorSearchResult(
            chunk_ids=chunk_ids,
            chunk_scores=chunk_scores,
            community_ids=community_ids,
            community_scores=community_scores,
            query_embedding=query_embedding,
        )

    def search_chunks_only(
        self,
        query: str,
        limit: int = 10,
        text_ids: set[int] | None = None,
    ) -> tuple[list[str], list[float], list[float]]:
        """
        Search only chunks (fallback if communities not built).
        
        Args:
            query: Search query text
            limit: Max chunks to return
            text_ids: If provided, restrict chunks to those from these text_ids (scope filter)
        
        Returns:
            (chunk_ids, scores, query_embedding)
        """
        query_embedding = self.get_embedding(query)
        results = self.storage.vector_search_chunks(
            query_embedding, limit=limit, text_ids=text_ids
        )
        chunk_ids = [r[0] for r in results]
        scores = [r[1] for r in results]
        return chunk_ids, scores, query_embedding

    def search_community_reports(
        self,
        query: str,
        limit: int = 5,
        text_ids: set[int] | None = None,
    ) -> "CommunityReportSearchResult":
        """
        Search community reports for GraphRAG routing.
        
        Args:
            query: Search query
            limit: Max communities to return
            text_ids: If provided, filter cited chunks to those from these text_ids (scope filter)
        
        Returns:
            CommunityReportSearchResult with community IDs, scores, and cited chunks.
            When text_ids provided, cited_chunk_ids are pre-filtered at SQL level.
        """
        query_embedding = self.get_embedding(query)
        results = self.storage.vector_search_community_reports(
            query_embedding, limit=limit, text_ids=text_ids
        )
        
        community_ids = []
        community_scores = []
        all_cited_chunks = []
        
        for row in results:
            comm_id, score, cited_chunks = row
            community_ids.append(comm_id)
            community_scores.append(score)
            if cited_chunks:
                all_cited_chunks.extend(cited_chunks)
        
        return CommunityReportSearchResult(
            community_ids=community_ids,
            scores=community_scores,
            cited_chunk_ids=list(set(all_cited_chunks)),
            query_embedding=query_embedding,
        )
