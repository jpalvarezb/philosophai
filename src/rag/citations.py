"""Build and format citations for LLM responses."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
    from ..schema import Citation


class CitationBuilder:
    """Build citation data for UI hover previews."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        node_to_community: dict[str, int] | None = None,
    ):
        self.storage = storage
        self.node_to_community = node_to_community or {}

    def build_citations(
        self,
        chunk_ids: list[str],
    ) -> list["Citation"]:
        """
        Build Citation objects for a list of chunk IDs.
        
        Args:
            chunk_ids: Ordered list of chunk IDs used in the answer
        
        Returns:
            List of Citation objects with index, content, and metadata
        """
        from ..schema import Citation

        chunks = self.storage.get_chunk_texts(chunk_ids)
        chunk_map = {c[0]: c[1] for c in chunks}

        # Map chunk -> provenance metadata (author/title/tradition)
        provenance = self.storage.get_chunk_provenance(chunk_ids)

        # Get entity associations for each chunk
        chunk_entities = self._get_chunk_entities(chunk_ids)

        citations = []
        for idx, chunk_id in enumerate(chunk_ids, start=1):
            content = chunk_map.get(chunk_id, "")
            entities = chunk_entities.get(chunk_id, [])
            meta = provenance.get(chunk_id) or provenance.get(str(chunk_id)) or {}
            author = meta.get("author")
            title = meta.get("title")
            tradition = meta.get("tradition")

            # Find community for this chunk (use first entity's community)
            community_id = None
            for entity_id in entities:
                if entity_id in self.node_to_community:
                    community_id = self.node_to_community[entity_id]
                    break

            citations.append(
                Citation(
                    index=idx,
                    chunk_id=chunk_id,
                    chunk_content=content,
                    community_id=community_id,
                    node_ids=entities,
                    author=author,
                    work=title,
                    title=title,
                    tradition=tradition,
                )
            )

        return citations

    def _get_chunk_entities(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """Get entity IDs mentioned in each chunk."""
        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT chunk_id, subject_canon_id, object_canon_id
            FROM normalized_triples_clean_canon
            WHERE chunk_id IN ({placeholders})
        """
        results = self.storage.con.execute(sql, chunk_ids).fetchall()

        chunk_entities: dict[str, set[str]] = {}
        for row in results:
            chunk_id, subj, obj = row
            if chunk_id not in chunk_entities:
                chunk_entities[chunk_id] = set()
            if subj:
                chunk_entities[chunk_id].add(subj)
            if obj:
                chunk_entities[chunk_id].add(obj)

        return {k: list(v) for k, v in chunk_entities.items()}

    def format_context_for_llm(
        self,
        chunks: list[tuple[str, str]],
    ) -> str:
        """
        Format chunks as numbered context for LLM prompt.
        
        Args:
            chunks: List of (chunk_id, content) tuples
        
        Returns:
            Formatted string with numbered citations
        """
        lines = []
        for idx, (chunk_id, content) in enumerate(chunks, start=1):
            lines.append(f"[{idx}] {content}")
        return "\n\n".join(lines)

    def to_dict_list(self, citations: list["Citation"]) -> list[dict]:
        """Convert citations to list of dicts for JSON response."""
        return [c.to_dict() for c in citations]
