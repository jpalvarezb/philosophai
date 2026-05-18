"""Corpus audit agent for ingestion parameter discovery."""

from __future__ import annotations

import json
import logging
import re
from statistics import mean
from typing import TYPE_CHECKING, Any

from ..config import DEFAULT_ENTITY_TYPES, CorpusAuditReport, IngestConfig

if TYPE_CHECKING:
    from openai import OpenAI

    from ...storage import DuckDBStorage


logger = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 200
_MAX_CHUNK_CHARS = 5000
_MIN_OVERLAP = 0
_MAX_OVERLAP = 1000
_VALID_CHUNK_METHODS = {"paragraph", "semantic"}
_GENERIC_SUBTYPE_TOKENS = {
    "analyst",
    "approach",
    "commentary",
    "concept",
    "dialogue",
    "doctrine",
    "ethicist",
    "framework",
    "literary",
    "logician",
    "method",
    "metaphysician",
    "movement",
    "mystic",
    "order",
    "philosopher",
    "philosophical",
    "poet",
    "practice",
    "school",
    "scholar",
    "scripture",
    "sect",
    "subtype",
    "sutra",
    "system",
    "text",
    "theologian",
    "theory",
    "tradition",
    "treatise",
    "work",
}

AUDIT_SYSTEM_PROMPT = """\
You are a corpus-profiling agent for a knowledge-graph ingestion pipeline.
Your job is to recommend conservative chunking parameters and identify
boilerplate/noise patterns that should be stripped before further processing.

Return a single JSON object with EXACTLY these keys:
{
  "chunk_max_chars": <int, 200-5000, default 1800>,
  "chunk_overlap": <int, 0-1000, default 300>,
  "chunk_method": "paragraph" | "semantic",
  "boilerplate_patterns": [<list of Python regex strings to strip from raw text>],
  "discovered_subtypes": {<entity_type>: [<subtype_labels>]},
  "findings": [<short human-readable observations about the corpus>]
}

Guidelines:
- Prefer "paragraph" chunking unless most documents lack clear paragraph breaks.
- boilerplate_patterns should be Python-compatible regex (re module).
  Target headers, footers, legal notices, and corpus-sourcing metadata.
  Keep patterns specific — avoid broad patterns that could eat real content.
- discovered_subtypes must map to one of the base entity types provided.
  These are domain-specific CATEGORY LABELS (e.g. "Ancient Greek Philosopher",
  "Ethical Theory", "Sacred Text"), NOT specific entity instances.
  WRONG: {"Person": ["Plato", "Aristotle"]}
  RIGHT: {"Person": ["Ancient Greek Philosopher", "Scholastic Theologian"]}
- If unsure about a recommendation, keep the default value.
- DO NOT invent data. Only report what the profiles evidence."""


class CorpusAuditAgent:
    """Profile the raw corpus and suggest chunking/boilerplate settings."""

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

    def collect_profiles(self, sample_limit: int = 12) -> dict[str, Any]:
        """Collect compact corpus statistics for the LLM."""
        con = self.storage.con

        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        if "files" not in tables or "raw_texts" not in tables:
            logger.warning("files/raw_texts tables missing — returning empty profiles")
            return {
                "document_count": 0,
                "format_counts": {},
                "char_stats": {},
                "paragraph_stats": {},
                "samples": [],
            }

        rows = con.execute("""
            SELECT f.text_id, f.title, f.author_source, f.file_ext, r.content
            FROM files f
            INNER JOIN raw_texts r ON f.text_id = r.text_id
            ORDER BY f.text_id
            """).fetchall()

        if not rows:
            return {
                "document_count": 0,
                "format_counts": {},
                "char_stats": {},
                "paragraph_stats": {},
                "samples": [],
            }

        lengths = [len(row[4] or "") for row in rows]
        paragraph_counts = [
            len([p for p in (row[4] or "").split("\n\n") if p.strip()]) for row in rows
        ]
        format_counts: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            ext = row[3] or "unknown"
            format_counts[ext] = format_counts.get(ext, 0) + 1
            if idx < sample_limit:
                content = row[4] or ""
                samples.append(
                    {
                        "text_id": row[0],
                        "title": row[1],
                        "author": row[2],
                        "file_ext": ext,
                        "prefix": content[:500],
                        "suffix": content[-500:] if len(content) > 500 else content,
                    }
                )

        return {
            "document_count": len(rows),
            "format_counts": format_counts,
            "char_stats": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": int(mean(lengths)),
            },
            "paragraph_stats": {
                "min": min(paragraph_counts),
                "max": max(paragraph_counts),
                "mean": round(mean(paragraph_counts), 2),
            },
            "samples": samples,
        }

    def _call_llm(self, profiles: dict[str, Any]) -> dict[str, Any]:
        """Ask the model to suggest chunking and boilerplate rules."""
        if self.llm_client is None:
            logger.info("No LLM client — returning default audit suggestions")
            return {}

        entity_types_summary = ", ".join(self.config.entity_types.keys())
        user_prompt = (
            f"Analyse the corpus profiles below and return your recommendations.\n"
            f"Valid base entity types for discovered_subtypes: {entity_types_summary}\n"
            f"Current defaults — chunk_max_chars: {self.config.chunk_max_chars}, "
            f"chunk_overlap: {self.config.chunk_overlap}, chunk_method: {self.config.chunk_method}\n\n"
            f"Profiles:\n{json.dumps(profiles, indent=2)}"
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            logger.error("Corpus audit LLM call failed: %s — using defaults", exc)
            return {}

    @staticmethod
    def _validate_regex_patterns(patterns: list) -> list[str]:
        """Keep only patterns that compile as valid Python regexes."""
        valid: list[str] = []
        for raw in patterns:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                re.compile(raw)
                valid.append(raw)
            except re.error as exc:
                logger.warning("Dropping invalid boilerplate regex %r: %s", raw, exc)
        return valid

    @staticmethod
    def _is_category_label(value: str) -> bool:
        """Return True only for generic subtype labels, not entity instances."""
        label = value.strip()
        if not label:
            return False

        normalized = re.sub(r"\s+", " ", label)
        tokens = [
            token.strip(".,:;!?()[]{}\"'").lower() for token in normalized.split()
        ]
        tokens = [token for token in tokens if token]
        if not tokens:
            return False

        if normalized.lower() in {base.lower() for base in DEFAULT_ENTITY_TYPES}:
            return False
        if len(normalized) < 4 or len(tokens) > 6:
            return False
        if len(tokens) == 1:
            return tokens[0] in _GENERIC_SUBTYPE_TOKENS
        return any(token in _GENERIC_SUBTYPE_TOKENS for token in tokens)

    def _clamp_suggestions(self, suggestions: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp LLM-suggested values to sane ranges."""
        clamped = dict(suggestions)

        if "chunk_max_chars" in clamped:
            try:
                val = int(clamped["chunk_max_chars"])
                clamped["chunk_max_chars"] = max(
                    _MIN_CHUNK_CHARS, min(val, _MAX_CHUNK_CHARS)
                )
            except (TypeError, ValueError):
                del clamped["chunk_max_chars"]

        if "chunk_overlap" in clamped:
            try:
                val = int(clamped["chunk_overlap"])
                max_chars = clamped.get("chunk_max_chars", self.config.chunk_max_chars)
                clamped["chunk_overlap"] = max(
                    _MIN_OVERLAP, min(val, _MAX_OVERLAP, max_chars // 2)
                )
            except (TypeError, ValueError):
                del clamped["chunk_overlap"]

        if "chunk_method" in clamped:
            if clamped["chunk_method"] not in _VALID_CHUNK_METHODS:
                logger.warning(
                    "LLM suggested invalid chunk_method %r — keeping default",
                    clamped["chunk_method"],
                )
                del clamped["chunk_method"]

        clamped["boilerplate_patterns"] = self._validate_regex_patterns(
            list(clamped.get("boilerplate_patterns", []))
        )

        valid_types = set(self.config.entity_types)
        raw_subtypes = dict(clamped.get("discovered_subtypes", {}))
        validated_subtypes: dict[str, list[str]] = {}
        for base_type, subtypes in raw_subtypes.items():
            if base_type not in valid_types:
                logger.warning(
                    "Dropping discovered_subtypes for unknown base type %r", base_type
                )
                continue
            if isinstance(subtypes, list):
                kept = [
                    str(s) for s in subtypes if s and self._is_category_label(str(s))
                ]
                dropped = [
                    str(s)
                    for s in subtypes
                    if s and not self._is_category_label(str(s))
                ]
                if dropped:
                    logger.warning(
                        "Dropped %d entity instances posing as subtypes under %s: %s",
                        len(dropped),
                        base_type,
                        dropped,
                    )
                if kept:
                    validated_subtypes[base_type] = kept
        clamped["discovered_subtypes"] = validated_subtypes

        return clamped

    def run(self) -> tuple[IngestConfig, CorpusAuditReport]:
        """Profile the corpus and return an updated config."""
        logger.info("Starting corpus audit …")
        profiles = self.collect_profiles()
        logger.info(
            "Corpus profile: %d documents, %s formats",
            profiles.get("document_count", 0),
            profiles.get("format_counts", {}),
        )

        raw_suggestions = self._call_llm(profiles)
        suggestions = self._clamp_suggestions(raw_suggestions)

        report = CorpusAuditReport(
            chunk_recommendations={
                "chunk_max_chars": suggestions.get(
                    "chunk_max_chars", self.config.chunk_max_chars
                ),
                "chunk_overlap": suggestions.get(
                    "chunk_overlap", self.config.chunk_overlap
                ),
                "chunk_method": suggestions.get(
                    "chunk_method", self.config.chunk_method
                ),
            },
            boilerplate_patterns=suggestions.get("boilerplate_patterns", []),
            discovered_subtypes=suggestions.get("discovered_subtypes", {}),
            sample_findings=list(suggestions.get("findings", [])),
            raw_profiles=profiles,
        )

        logger.info("Audit recommendations: %s", report.chunk_recommendations)
        if report.boilerplate_patterns:
            logger.info(
                "Discovered %d boilerplate patterns", len(report.boilerplate_patterns)
            )
        if report.discovered_subtypes:
            logger.info("Discovered subtypes: %s", report.discovered_subtypes)

        updated = self.config.merge_overrides(
            chunk_max_chars=report.chunk_recommendations.get(
                "chunk_max_chars", self.config.chunk_max_chars
            ),
            chunk_overlap=report.chunk_recommendations.get(
                "chunk_overlap", self.config.chunk_overlap
            ),
            chunk_method=report.chunk_recommendations.get(
                "chunk_method", self.config.chunk_method
            ),
        )
        updated.boilerplate_patterns = (
            report.boilerplate_patterns or updated.boilerplate_patterns
        )
        updated.discovered_subtypes = (
            report.discovered_subtypes or updated.discovered_subtypes
        )
        updated.audit_report = report
        return updated, report
