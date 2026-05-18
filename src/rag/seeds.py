"""Query-aware seed entity selection for traversal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..config.logging import trace_logger, TRACE_VERBOSE, TRACE_MAX_ITEMS

if TYPE_CHECKING:
    import networkx as nx
    from ..storage import DuckDBStorage
    from ..graph.filters import GraphFilters


# Query stopwords to exclude from matching
QUERY_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "about",
        "against",
        "its",
        "our",
        "their",
        "your",
        "my",
        "his",
        "her",
        "according",
        "relationship",
        "between",
        "explain",
        "describe",
        "discuss",
        "compare",
        "contrast",
    }
)


@dataclass
class ScoredEntity:
    """Entity with relevance score for seeding."""

    entity_id: str
    label: str
    score: float
    community_id: int | None
    reason: str  # Why this entity was selected


def tokenize_query(query: str) -> set[str]:
    """Extract meaningful tokens from query."""
    tokens = re.findall(r"[a-z]+", query.lower())
    return {t for t in tokens if len(t) > 2 and t not in QUERY_STOPWORDS}


def score_label_relevance(label: str, query_tokens: set[str]) -> tuple[float, str]:
    """
    Score entity label relevance to query tokens.

    Returns:
        (score, reason) tuple
    """
    label_tokens = set(re.findall(r"[a-z]+", label.lower()))

    if not label_tokens or not query_tokens:
        return 0.0, ""

    # Exact token overlap
    overlap = label_tokens & query_tokens
    if overlap:
        score = len(overlap) / min(len(label_tokens), len(query_tokens))
        return score, f"exact:{','.join(overlap)}"

    # Substring matching (partial)
    for qt in query_tokens:
        for lt in label_tokens:
            # Query token is substring of label token
            if len(qt) >= 4 and qt in lt:
                return 0.4, f"partial:{qt}⊂{lt}"
            # Label token is substring of query token
            if len(lt) >= 4 and lt in qt:
                return 0.3, f"partial:{lt}⊂{qt}"

    return 0.0, ""


def score_entities_for_query(
    query: str,
    candidate_entity_ids: list[str],
    graph: "nx.MultiDiGraph",
    node_to_community: dict[str, int],
    target_communities: list[int],
    filters: "GraphFilters",
    max_results: int = 30,
) -> list[ScoredEntity]:
    """
    Score and rank candidate entities by relevance to query.

    Args:
        query: User question
        candidate_entity_ids: Entity IDs from chunk extraction
        graph: Knowledge graph for labels
        node_to_community: Mapping of entity -> community
        target_communities: Communities from report routing
        filters: GraphFilters instance for validation
        max_results: Maximum entities to return

    Returns:
        List of ScoredEntity, sorted by score descending
    """
    query_tokens = tokenize_query(query)
    trace_logger.decision(f"seed_score tokens={sorted(query_tokens)[:TRACE_MAX_ITEMS]}")
    target_set = set(target_communities)

    scored: list[ScoredEntity] = []

    for entity_id in candidate_entity_ids:
        # Must pass filter validation
        if not filters.is_valid_seed(entity_id):
            continue

        # Get label
        label = graph.nodes.get(entity_id, {}).get("label", entity_id)

        # Score by query relevance
        relevance, reason = score_label_relevance(label, query_tokens)

        # Community bonus
        comm_id = node_to_community.get(entity_id)
        community_bonus = 0.0
        if comm_id in target_set:
            community_bonus = 0.3
            if reason:
                reason += ",comm_match"
            else:
                reason = "comm_match"

        # Combined score
        score = relevance + community_bonus

        # Include if relevant OR in target community
        if score > 0:
            scored.append(
                ScoredEntity(
                    entity_id=entity_id,
                    label=label,
                    score=score,
                    community_id=comm_id,
                    reason=reason,
                )
            )

    # Sort by score descending
    scored.sort(key=lambda x: x.score, reverse=True)
    if TRACE_VERBOSE:
        trace_logger.score(
            f"seed_score_top={[(s.label, round(s.score, 3), s.reason) for s in scored[:TRACE_MAX_ITEMS]]}"
        )

    return scored[:max_results]


def select_seeds(
    query: str,
    chunk_ids: list[str],
    storage: "DuckDBStorage",
    graph: "nx.MultiDiGraph",
    node_to_community: dict[str, int],
    target_communities: list[int],
    filters: "GraphFilters",
    max_seeds: int = 20,
) -> list[ScoredEntity]:
    """
    Convenience function: extract entities from chunks and score them.

    This is the main entry point for seed selection.
    """
    # Get candidate entities from chunks
    candidates = storage.get_entity_ids_from_chunks(chunk_ids)
    trace_logger.decision(
        f"seed_candidates chunks={len(chunk_ids)} candidates={len(candidates)}"
    )

    # Score and filter
    scored = score_entities_for_query(
        query=query,
        candidate_entity_ids=candidates,
        graph=graph,
        node_to_community=node_to_community,
        target_communities=target_communities,
        filters=filters,
        max_results=max_seeds,
    )

    return scored
