"""Dynamic cleaning agent for extracted triples."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from ..config import CleaningRules, IngestConfig

if TYPE_CHECKING:
    from openai import OpenAI

    from ...storage import DuckDBStorage


logger = logging.getLogger(__name__)

DEFAULT_PATTERN_PROBES = [
    "%gutenberg%",
    "%chapter%",
    "%copyright%",
    "%license%",
    "%ebook%",
    "%preface%",
    "%introduction%",
]

_PROVENANCE_SCAN_LIMIT = 50_000

_LIKE_PATTERN_RE = re.compile(r"^[%_a-zA-Z0-9 .\-]+$")

CLEANING_SYSTEM_PROMPT = """\
You are a cleaning-rules agent for a knowledge-graph ingestion pipeline.
Your job is to propose conservative SQL LIKE patterns to remove noise entities
and predicates that do NOT represent real philosophical or domain knowledge.

Return a single JSON object with EXACTLY these keys:
{
  "entity_patterns": ["<SQL LIKE pattern>", ...],
  "predicate_patterns": ["<SQL LIKE pattern>", ...],
  "type_rules": {"<EntityType>": ["<SQL LIKE pattern>", ...]},
  "explanations": {"<pattern>": "<why this pattern is noise>"}
}

Rules for SQL LIKE patterns:
- Use lowercase — comparison is done with LOWER().
- % matches any sequence; _ matches a single character.
- Examples: "%gutenberg%", "%copyright%", "%ebook%"
- Only propose a pattern when the profile evidence is strong (high frequency,
  clearly non-domain content like legal boilerplate, corpus metadata, UI elements).
- NEVER propose patterns that would match real philosophical entities or concepts.
- type_rules keys must be one of the valid entity types provided.
- Keep it conservative — false negatives are far better than false positives."""

VALIDATION_SYSTEM_PROMPT = """\
You are reviewing a candidate SQL LIKE noise-filter pattern for a philosophy
knowledge graph. Given the pattern and sample entities it would remove,
decide whether the pattern is safe to keep.

Return JSON: {"keep": true/false, "rationale": "<reason>"}

Keep the pattern ONLY if the matched entities are clearly non-philosophical
noise (corpus metadata, legal text, UI elements, boilerplate). Reject if any
matched entity looks like a legitimate philosophical concept, person, or work."""


class CleaningAgent:
    """Profile extracted triples and propose conservative cleaning rules."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI | None" = None,
        config: IngestConfig | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.config = config or IngestConfig()
        self.model = model

    def _table_exists(self, name: str) -> bool:
        tables = [row[0] for row in self.storage.con.execute("SHOW TABLES").fetchall()]
        return name in tables

    def collect_profiles(self) -> dict[str, Any]:
        """Collect triple/type distribution stats for the LLM."""
        if not self._table_exists("normalized_triples"):
            logger.warning(
                "normalized_triples table missing — returning empty profiles"
            )
            return {"total_triples": 0}

        con = self.storage.con
        total = con.execute("SELECT COUNT(*) FROM normalized_triples").fetchone()[0]
        if total == 0:
            return {"total_triples": 0}

        top_subjects = con.execute("""
            SELECT subject_norm, COUNT(*) AS freq
            FROM normalized_triples
            GROUP BY 1
            ORDER BY freq DESC
            LIMIT 25
            """).fetchall()
        top_predicates = con.execute("""
            SELECT predicate_norm, COUNT(*) AS freq
            FROM normalized_triples
            GROUP BY 1
            ORDER BY freq DESC
            LIMIT 25
            """).fetchall()
        type_counts = con.execute("""
            SELECT subject_type, COUNT(*) AS freq
            FROM normalized_triples
            GROUP BY 1
            ORDER BY freq DESC
            """).fetchall()

        pattern_counts: dict[str, int] = {}
        for pattern in DEFAULT_PATTERN_PROBES:
            pattern_counts[pattern] = con.execute(
                """
                SELECT COUNT(*)
                FROM normalized_triples
                WHERE LOWER(subject_norm) LIKE ?
                   OR LOWER(object_norm) LIKE ?
                """,
                [pattern, pattern],
            ).fetchone()[0]

        short_people = con.execute("""
            SELECT DISTINCT subject_norm
            FROM normalized_triples
            WHERE subject_type = 'Person'
              AND LENGTH(subject_norm) <= 3
            LIMIT 25
            """).fetchall()

        return {
            "total_triples": total,
            "top_subjects": [
                {"entity": row[0], "freq": row[1]} for row in top_subjects
            ],
            "top_predicates": [
                {"predicate": row[0], "freq": row[1]} for row in top_predicates
            ],
            "type_counts": [{"type": row[0], "freq": row[1]} for row in type_counts],
            "pattern_counts": pattern_counts,
            "short_people": [row[0] for row in short_people],
        }

    def find_provenance_failures(self) -> list[str]:
        """Return entities that repeatedly fail source-text provenance checks."""
        if not self._table_exists("entity_chunks") or not self._table_exists("chunks"):
            logger.warning(
                "entity_chunks/chunks tables missing — skipping provenance scan"
            )
            return []

        try:
            from rapidfuzz import fuzz
        except ImportError:  # pragma: no cover - dependency-driven
            logger.warning("rapidfuzz not installed — skipping provenance verification")
            return []

        rows = self.storage.con.execute(
            """
            SELECT ec.entity_norm, ec.chunk_id, c.content
            FROM entity_chunks ec
            INNER JOIN chunks c ON ec.chunk_id = c.chunk_id
            LIMIT ?
            """,
            [_PROVENANCE_SCAN_LIMIT],
        ).fetchall()
        logger.info("Provenance scan: checking %d entity-chunk pairs", len(rows))

        total_by_entity: Counter[str] = Counter()
        failed_by_entity: Counter[str] = Counter()
        for entity, _, content in rows:
            if not entity or not content:
                continue
            total_by_entity[entity] += 1
            haystack = content.lower()
            needle = entity.lower()
            if needle in haystack:
                continue
            score = fuzz.partial_ratio(needle, haystack) / 100.0
            if score < self.config.provenance_threshold:
                failed_by_entity[entity] += 1

        flagged = [
            entity
            for entity, failed in failed_by_entity.items()
            if total_by_entity[entity] >= 2 and failed / total_by_entity[entity] > 0.5
        ]
        logger.info(
            "Provenance failures: %d entities flagged out of %d checked",
            len(flagged),
            len(total_by_entity),
        )
        return sorted(flagged)

    @staticmethod
    def _is_valid_like_pattern(pattern: str) -> bool:
        """Check that a string looks like a safe SQL LIKE pattern."""
        if not isinstance(pattern, str) or not pattern.strip():
            return False
        if "'" in pattern or ";" in pattern or "--" in pattern:
            return False
        return bool(_LIKE_PATTERN_RE.match(pattern.strip()))

    def _call_llm(
        self, profiles: dict[str, Any], provenance_failed_entities: list[str]
    ) -> dict[str, Any]:
        if self.llm_client is None:
            logger.info("No LLM client — returning empty cleaning suggestions")
            return {}

        entity_types = ", ".join(self.config.entity_types.keys())
        user_prompt = (
            f"Valid entity types for type_rules: {entity_types}\n\n"
            f"Triple profiles:\n{json.dumps(profiles, indent=2)}\n\n"
            f"Provenance-failed entities (top 100):\n"
            f"{json.dumps(provenance_failed_entities[:100], indent=2)}"
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            logger.error("Cleaning agent LLM call failed: %s — using empty rules", exc)
            return {}

    def _sample_matches(self, pattern: str, limit: int = 10) -> list[str]:
        if not self._table_exists("normalized_triples"):
            return []
        rows = self.storage.con.execute(
            """
            SELECT DISTINCT subject_norm
            FROM normalized_triples
            WHERE LOWER(subject_norm) LIKE ?
               OR LOWER(object_norm) LIKE ?
            LIMIT ?
            """,
            [pattern, pattern, limit],
        ).fetchall()
        return [row[0] for row in rows if row[0]]

    def validate_patterns(self, candidate_patterns: list[str]) -> list[str]:
        """Validate candidate patterns: format check, then LLM confirmation."""
        format_valid = [p for p in candidate_patterns if self._is_valid_like_pattern(p)]
        rejected_format = set(candidate_patterns) - set(format_valid)
        if rejected_format:
            logger.warning(
                "Rejected %d patterns with invalid SQL LIKE format: %s",
                len(rejected_format),
                rejected_format,
            )

        if self.llm_client is None:
            return format_valid

        confirmed: list[str] = []
        for pattern in format_valid:
            samples = self._sample_matches(pattern)
            if not samples:
                logger.debug(
                    "Pattern %r matched no samples — skipping validation", pattern
                )
                continue
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Pattern: {pattern}\n"
                                f"Sample matches: {json.dumps(samples, indent=2)}\n"
                                "Should this pattern be kept as a conservative noise filter?"
                            ),
                        },
                    ],
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                if payload.get("keep"):
                    confirmed.append(pattern)
                    logger.debug(
                        "Pattern %r confirmed: %s",
                        pattern,
                        payload.get("rationale", ""),
                    )
                else:
                    logger.info(
                        "Pattern %r rejected by LLM: %s",
                        pattern,
                        payload.get("rationale", ""),
                    )
            except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    "Validation LLM call failed for pattern %r: %s — keeping pattern",
                    pattern,
                    exc,
                )
                confirmed.append(pattern)
        return confirmed

    def run(self) -> tuple[IngestConfig, CleaningRules]:
        """Generate dynamic cleaning rules and merge them into config."""
        logger.info("Starting cleaning agent …")
        profiles = self.collect_profiles()
        if profiles.get("total_triples", 0) == 0:
            logger.warning("No triples found — returning empty cleaning rules")
            return self.config, CleaningRules()

        provenance_failed_entities = self.find_provenance_failures()
        suggestions = self._call_llm(profiles, provenance_failed_entities)

        raw_entity_patterns = list(suggestions.get("entity_patterns", []))
        predicate_patterns = [
            p
            for p in suggestions.get("predicate_patterns", [])
            if self._is_valid_like_pattern(p)
        ]
        validated_entity_patterns = self.validate_patterns(raw_entity_patterns)

        valid_types = set(self.config.entity_types)
        raw_type_rules = dict(suggestions.get("type_rules", {}))
        validated_type_rules: dict[str, list[str]] = {}
        for entity_type, patterns in raw_type_rules.items():
            if entity_type not in valid_types:
                logger.warning(
                    "Dropping type_rules for unknown entity type %r", entity_type
                )
                continue
            safe = [p for p in patterns if self._is_valid_like_pattern(p)]
            if safe:
                validated_type_rules[entity_type] = safe

        rules = CleaningRules(
            entity_patterns=validated_entity_patterns,
            predicate_patterns=predicate_patterns,
            provenance_failed_entities=provenance_failed_entities,
            type_rules=validated_type_rules,
            explanations=dict(suggestions.get("explanations", {})),
        )

        logger.info(
            "Cleaning rules: %d entity patterns, %d predicate patterns, %d provenance failures, %d type rules",
            len(rules.entity_patterns),
            len(rules.predicate_patterns),
            len(rules.provenance_failed_entities),
            len(rules.type_rules),
        )

        updated = self.config.merge_overrides(
            noise_entity_patterns=rules.entity_patterns,
            noise_predicate_patterns=rules.predicate_patterns,
        )
        updated.cleaning_rules = rules
        return updated, rules
