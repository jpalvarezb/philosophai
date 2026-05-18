"""Typed triple extraction with provenance tracking."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel, ValidationError

from .config import IngestConfig

if TYPE_CHECKING:
    from openai import OpenAI

    from ..storage import DuckDBStorage


logger = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
SINGLE_LETTER_RE = re.compile(r"^[A-Za-z]$")
HEADING_LINE_RE = re.compile(
    r"^(chapter|section|book|part|volume|appendix|index)\b|^[ivxlcdm]+\.$|^\d+([.)]\d+)*$",
    re.IGNORECASE,
)
CHAPTER_NUMERAL_RE = re.compile(
    r"(?:^|[\s_])(chapter|section|part|book|volume|appendix)[\s_]+([IVXLCDM]+|\d+)\b",
    re.IGNORECASE,
)
INDEX_GLOSSARY_RE = re.compile(
    r"\bsee\s+also\b|\bsee\s+under\b|\bcf\.\s",
    re.IGNORECASE,
)
PLACEHOLDER_ENTITY_RE = re.compile(
    r"^[_\s]*n(?:[._\s])*?$",
    re.IGNORECASE,
)

PRONOUN_TOKENS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "our",
    "their",
    "mine",
    "yours",
    "ours",
    "theirs",
    "myself",
    "yourself",
    "ourselves",
    "himself",
    "herself",
    "itself",
    "themselves",
}
STRUCTURAL_TOKENS = {
    "chapter",
    "section",
    "appendix",
    "index",
    "contents",
    "table of contents",
    "toc",
    "preface",
    "introduction",
}
LEGAL_MARKERS = (
    "project gutenberg",
    "copyright",
    "license",
    "terms of use",
    "all rights reserved",
    "refund",
    "warranty",
    "donation",
)


_SYSTEM_PROMPT_TEMPLATE = """\
You are an information extraction engine.
Task: extract concise factual subject-predicate-object triples from the provided chunks.

Return a single JSON object with this exact structure:
{{
  "chunks": [
    {{
      "chunk_id": "chunk-id",
      "triples": [
        {{
          "subject": "...",
          "subject_type": "<EntityType>",
          "predicate": "...",
          "object": "...",
          "object_type": "<EntityType>"
        }}
      ]
    }}
  ]
}}

Valid entity types (assign exactly one per entity):
{entity_type_block}

Examples:
- Example 1 (keep semantic claims):
  Text:
  "Horus is identified with the rising sun. The Bennu bird symbolizes rebirth."
  Output triples:
  - subject="Horus", subject_type="Person", predicate="is_identified_with", object="the rising sun", object_type="Concept"
  - subject="Bennu bird", subject_type="Concept", predicate="symbolizes", object="rebirth", object_type="Concept"

- Example 2 (reject structural/editorial references):
  Text:
  "Chapter XII is titled The Weighing of the Heart. Papyrus of Ani preserves this title."
  Output triples:
  - none (return empty triples list for this chunk)

- Example 3 (reject unresolved pronoun speaker):
  Text:
  "I am the one who opens the gates of heaven."
  Output triples:
  - none (the speaker is not explicitly resolved to a concrete entity in the chunk)

- Example 4 (reject vague "is" relation):
  Text:
  "The teaching is about order and law."
  Output triples:
  - none (the relation is too vague; no concrete semantic relation is expressed)

- Example 5 (reject publication metadata):
  Text:
  "The Rule of St. Benedict was translated into English by W.K. Lowther Clarke and published in London."
  Output triples:
  - none (this is publication metadata, not intellectual content)

Rules:
- Subjects and objects must be concise noun phrases of 1–8 words. Do not extract full sentences, quotations, relative clauses, or verb phrases as entities.
- Predicates must be short verb phrases in snake_case.
- When uncertain about type, prefer {fallback}.
- If a chunk has no usable triples, return an empty triples list for that chunk.
- Extract only factual, definitional, or argumentative claims about concepts, persons, works, methods, places, events, organizations, or technologies.
- Do NOT extract reader instructions, legal/license text, publication metadata, or structural references (chapter/section/table-of-contents/index lines).
- Do NOT extract triples about chapter structure, section numbering, manuscript organization, rubrics, editorial apparatus, or document layout. These describe the document's structure, not its intellectual content.
- Do NOT extract structural references such as Chapter/Section/Verse/Stanza/Folio numbering, title labels, or passage location markers.
- Never use "Chapter X", "Section Y", "Part Z", or similar structural labels as subjects or objects.
- Skip speculative or hedged claims (e.g., "might", "may", "perhaps", "seems", "appears", "likely", "unless perhaps").
- If the subject or object is an unresolved pronoun/speaker reference ("I", "he", "they", "the speaker"), skip the triple unless the entity is explicitly named in the same chunk.
- Prefer precise relational predicates grounded in explicit wording; avoid vague predicates like "is related to", "is about", "has", "says", or "is of" unless they are the only explicit factual relation.
- Never use pronouns or single letters as subjects or objects.
- Do not invent chunks or entities not grounded in the text."""

_EVALUATOR_SYSTEM_PROMPT = """\
You are a triple quality evaluator for a knowledge graph ingestion pipeline.
Task: review candidate subject-predicate-object triples extracted from source chunks and decide whether each triple should be kept.

Return a single JSON object with this exact structure:
{
  "triples": [
    {
      "triple_id": 0,
      "keep": true,
      "reason": "short explanation"
    }
  ]
}

Rules:
- Keep only triples that are explicitly grounded in the chunk text.
- Keep only triples that express semantic content, not document structure.
- Reject triples about chapter structure, section numbering, table of contents, indexes, page layout, editorial apparatus, publication metadata (e.g., translators, publishers, publication locations), or legal text.
- Reject numbered structural labels or headings used as entities, such as Chapter XII, Book III, Section 2, or Appendix B, when they refer to document structure rather than substantive content.
- Reject claims about where a passage appears, what a chapter is titled, what another manuscript version contains, or which copy preserves a title. This includes predicates like "is_known_in" or "is_opened_in" pointing to Chapter numbers.
- Do not reject a triple only because a work or manuscript is mentioned; keep it when the relation expresses substantive content rather than the text's organization or transmission history.
- Reject unresolved role entities such as "the speaker", "the narrator", "the author", or similar role labels when they are not explicitly resolved to a concrete named entity in the chunk.
- Reject triples whose subject or object is not a meaningful entity or concept in context.
- Reject triples with malformed, vague, or non-relational predicates.
- Prefer rejecting uncertain triples over keeping noisy ones.
- Return JSON only."""


def build_system_prompt(config: IngestConfig) -> str:
    """Build the extraction system prompt from the live entity type config."""
    lines = []
    for type_name, description in config.entity_types.items():
        lines.append(f"  - {type_name}: {description}")
    return _SYSTEM_PROMPT_TEMPLATE.format(
        entity_type_block="\n".join(lines),
        fallback=config.entity_type_fallback,
    )


def build_evaluator_prompt(candidates: Sequence[CandidateTriple]) -> str:
    """Build the user prompt for triple evaluation."""
    chunk_groups: dict[str, list[CandidateTriple]] = {}
    chunk_texts: dict[str, str] = {}
    for candidate in candidates:
        chunk_groups.setdefault(candidate.chunk_id, []).append(candidate)
        chunk_texts[candidate.chunk_id] = candidate.chunk_text.strip()

    sections = [
        f"Review {len(candidates)} candidate triples. Decide whether each one should be kept."
    ]
    for idx, (chunk_id, chunk_candidates) in enumerate(chunk_groups.items(), start=1):
        lines = [f"--- Chunk {idx} ---", f"Chunk ID: {chunk_id}", "Text:", chunk_texts[chunk_id], "", "Candidates:"]
        for candidate in chunk_candidates:
            lines.append(
                (
                    f"- Triple ID: {candidate.triple_id} | "
                    f"({candidate.subject_str} [{candidate.subject_type}]) --{candidate.predicate_str}--> "
                    f"({candidate.object_str} [{candidate.object_type}])"
                )
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


class TypedTriple(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str


class ChunkTriples(BaseModel):
    chunk_id: str
    triples: list[TypedTriple]


class BatchTriplesResponse(BaseModel):
    chunks: list[ChunkTriples]


class TripleEvaluation(BaseModel):
    triple_id: int
    keep: bool
    reason: str = ""


class TripleEvaluationsResponse(BaseModel):
    triples: list[TripleEvaluation]


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text_id: int
    content: str


@dataclass(slots=True)
class CandidateTriple:
    triple_id: int
    text_id: int
    chunk_id: str
    chunk_text: str
    subject_str: str
    subject_norm: str
    subject_type: str
    predicate_str: str
    predicate_norm: str
    object_str: str
    object_norm: str
    object_type: str


def normalize_entity(text: str) -> str:
    """Normalize an entity surface form without destroying casing."""
    cleaned = (text or "").strip().strip("\"'`[]()")
    cleaned = cleaned.replace("\n", " ")
    return WHITESPACE_RE.sub(" ", cleaned)


def normalize_predicate(text: str) -> str:
    """Normalize predicates into canonical space-separated strings."""
    cleaned = (text or "").strip().replace("_", " ").replace("-", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned


def clean_json_text(text: str) -> str:
    """Extract the JSON object from a model response."""
    stripped = text.strip()
    match = JSON_BLOCK_RE.search(stripped)
    if match:
        return match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1:
        return stripped[start : end + 1]
    return stripped


class TripleExtractor:
    """Extract typed triples from chunks and persist provenance."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI | None" = None,
        config: IngestConfig | None = None,
    ):
        self.storage = storage
        self.config = config or IngestConfig()
        self.llm_client = llm_client or self._build_client()

    def _build_client(self) -> "OpenAI":
        from openai import OpenAI

        api_key = os.environ.get(self.config.extraction_api_key_env)
        base_url = self.config.extraction_api_base or ""
        is_local_ollama = any(
            token in base_url.lower()
            for token in (
                "127.0.0.1:11434",
                "127.0.0.1",
                "localhost:11434",
                "localhost",
                "ollama.com",
            )
        )

        if not api_key:
            if "openai.com" in base_url.lower() or is_local_ollama:
                api_key = api_key or "ollama"
            else:
                raise ValueError(
                    f"{self.config.extraction_api_key_env} environment variable is required for extraction"
                )

        client_kwargs: dict[str, object] = {"api_key": api_key, "base_url": self.config.extraction_api_base}
        if self.config.extraction_http_timeout_seconds is not None:
            client_kwargs["timeout"] = float(self.config.extraction_http_timeout_seconds)
        return OpenAI(**client_kwargs)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format elapsed/remaining seconds for logs."""
        total_seconds = max(0, int(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def ensure_schema(self) -> None:
        """Create extraction tables and indexes if needed."""
        con = self.storage.con
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS normalized_triples (
                triple_id VARCHAR PRIMARY KEY,
                text_id INTEGER,
                chunk_id VARCHAR,
                subject_str VARCHAR,
                subject_norm VARCHAR,
                subject_type VARCHAR,
                predicate_str VARCHAR,
                predicate_norm VARCHAR,
                object_str VARCHAR,
                object_norm VARCHAR,
                object_type VARCHAR,
                model_name VARCHAR,
                run_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_normalized_triples_chunk_id ON normalized_triples(chunk_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_normalized_triples_text_id ON normalized_triples(text_id)")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_chunks (
                entity_norm VARCHAR,
                entity_type VARCHAR,
                chunk_id VARCHAR,
                text_id INTEGER,
                role VARCHAR,
                PRIMARY KEY (entity_norm, chunk_id, role)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_entity_chunks_chunk_id ON entity_chunks(chunk_id)")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_types (
                entity_norm VARCHAR,
                entity_type VARCHAR,
                subtype VARCHAR,
                mention_count INTEGER,
                PRIMARY KEY (entity_norm, entity_type)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_skips (
                chunk_id VARCHAR PRIMARY KEY,
                reason VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_extraction_skips_reason ON extraction_skips(reason)")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_rejections (
                rejection_id VARCHAR PRIMARY KEY,
                text_id INTEGER,
                chunk_id VARCHAR,
                subject_str VARCHAR,
                subject_norm VARCHAR,
                predicate_str VARCHAR,
                predicate_norm VARCHAR,
                object_str VARCHAR,
                object_norm VARCHAR,
                stage VARCHAR,
                reason VARCHAR,
                model_name VARCHAR,
                run_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_extraction_rejections_chunk_id ON extraction_rejections(chunk_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_extraction_rejections_stage ON extraction_rejections(stage)")

    def fetch_pending_chunks(
        self,
        limit: int | None = None,
        levels: Sequence[str] = ("local",),
    ) -> list[ChunkRecord]:
        """Return chunks that have not yet been extracted."""
        con = self.storage.con
        level_placeholders = ",".join(["?"] * len(levels))
        params: list[object] = list(levels)
        query = f"""
            SELECT c.chunk_id, c.text_id, c.content
            FROM chunks c
            WHERE c.level IN ({level_placeholders})
              AND c.chunk_id NOT IN (SELECT DISTINCT chunk_id FROM normalized_triples)
              AND c.chunk_id NOT IN (SELECT DISTINCT chunk_id FROM extraction_skips)
            ORDER BY c.chunk_id
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = con.execute(query, params).fetchall()
        return [ChunkRecord(chunk_id=row[0], text_id=row[1], content=row[2]) for row in rows]

    def count_pending_chunks(self, levels: Sequence[str] = ("local",)) -> int:
        """Count chunks that have not yet been extracted."""
        con = self.storage.con
        level_placeholders = ",".join(["?"] * len(levels))
        query = f"""
            SELECT COUNT(*)
            FROM chunks c
            WHERE c.level IN ({level_placeholders})
              AND c.chunk_id NOT IN (SELECT DISTINCT chunk_id FROM normalized_triples)
              AND c.chunk_id NOT IN (SELECT DISTINCT chunk_id FROM extraction_skips)
        """
        return int(con.execute(query, list(levels)).fetchone()[0])

    @staticmethod
    def _chunk_skip_reason(content: str) -> str | None:
        """Return reason to skip chunk extraction if obviously structural/noise."""
        if not content or not content.strip():
            return "empty_chunk"

        lower = content.lower()
        if any(marker in lower for marker in LEGAL_MARKERS):
            return "legal_boilerplate_chunk"

        if "table of contents" in lower:
            return "table_of_contents_chunk"

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "empty_chunk"

        sample_lines = lines[:80]
        n_sample = max(1, len(sample_lines))

        heading_like = sum(1 for line in sample_lines if HEADING_LINE_RE.match(line))
        heading_ratio = heading_like / n_sample
        if heading_like >= 8 and heading_ratio >= 0.5:
            return "structural_heading_chunk"

        chapter_refs = sum(1 for line in sample_lines if CHAPTER_NUMERAL_RE.search(line))
        if chapter_refs >= 5 and chapter_refs / n_sample >= 0.3:
            return "toc_like_chapter_listing"

        idx_hits = sum(1 for line in sample_lines if INDEX_GLOSSARY_RE.search(line))
        if idx_hits >= 3 and idx_hits / n_sample >= 0.15:
            return "index_glossary_chunk"

        tokens = content.split()
        if len(tokens) <= 50 and heading_ratio >= 0.6:
            return "short_header_only_chunk"

        return None

    def insert_chunk_skips(self, skip_rows: list[tuple[str, str]]) -> None:
        """Persist chunk skip decisions so skipped chunks are not retried."""
        if not skip_rows:
            return
        self.storage.con.executemany(
            """
            INSERT INTO extraction_skips (chunk_id, reason)
            VALUES (?, ?)
            ON CONFLICT (chunk_id) DO UPDATE SET reason = EXCLUDED.reason
            """,
            skip_rows,
        )

    @staticmethod
    def _triple_rejection_reason(subject_norm: str, predicate_norm: str, object_norm: str) -> str | None:
        """Return reason to reject low-quality triples."""
        s = (subject_norm or "").strip()
        p = (predicate_norm or "").strip()
        o = (object_norm or "").strip()
        s_lower = s.lower()
        o_lower = o.lower()
        p_lower = p.lower()

        if s_lower in PRONOUN_TOKENS:
            return "pronoun_subject"
        if o_lower in PRONOUN_TOKENS:
            return "pronoun_object"
        if s_lower in STRUCTURAL_TOKENS:
            return "structural_subject"
        if o_lower in STRUCTURAL_TOKENS:
            return "structural_object"
        if PLACEHOLDER_ENTITY_RE.match(s_lower):
            return "placeholder_subject"
        if PLACEHOLDER_ENTITY_RE.match(o_lower):
            return "placeholder_object"
        if SINGLE_LETTER_RE.match(s):
            return "single_letter_subject"
        if SINGLE_LETTER_RE.match(o):
            return "single_letter_object"
        if s.isascii() and s.isalpha() and len(s) <= 2:
            return "very_short_ascii_subject"
        if o.isascii() and o.isalpha() and len(o) <= 2:
            return "very_short_ascii_object"

        MAX_ENTITY_WORDS = 8
        if len(s.split()) > MAX_ENTITY_WORDS:
            return "entity_too_long_subject"
        if len(o.split()) > MAX_ENTITY_WORDS:
            return "entity_too_long_object"

        combined = f"{s_lower} {p_lower} {o_lower}"
        if any(marker in combined for marker in LEGAL_MARKERS):
            return "legal_boilerplate_triple"
        return None

    def build_prompt(self, batch: Sequence[ChunkRecord]) -> str:
        """Build the user prompt for a chunk batch."""
        sections = []
        for idx, record in enumerate(batch, start=1):
            sections.append(f"--- Chunk {idx} ---\nID: {record.chunk_id}\nText:\n{record.content.strip()}")
        types = ", ".join(self.config.entity_types.keys())
        return (
            f"Process these {len(batch)} chunks and return the JSON object.\n"
            f"Valid entity types: {types}. Prefer {self.config.entity_type_fallback} when uncertain.\n\n"
            + "\n\n".join(sections)
        )

    def _completion_options(self, model_name: str) -> dict[str, object]:
        """Return model-compatible chat completion options."""
        options: dict[str, object] = {
            "response_format": {"type": "json_object"},
        }
        if not model_name.startswith("gpt-5"):
            options["temperature"] = 0.1
        return options

    def extract_batch(self, batch: Sequence[ChunkRecord]) -> dict[str, list[TypedTriple]]:
        """Run one model call for a chunk batch."""
        if not batch:
            return {}

        response = self.llm_client.chat.completions.create(
            model=self.config.extraction_model,
            messages=[
                {"role": "system", "content": build_system_prompt(self.config)},
                {"role": "user", "content": self.build_prompt(batch)},
            ],
            **self._completion_options(self.config.extraction_model.lower()),
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(clean_json_text(content))
        parsed = BatchTriplesResponse.model_validate(payload)
        valid_types = set(self.config.entity_types)

        result: dict[str, list[TypedTriple]] = {}
        for chunk in parsed.chunks:
            triples: list[TypedTriple] = []
            for triple in chunk.triples:
                if triple.subject_type not in valid_types:
                    triple.subject_type = self.config.entity_type_fallback
                if triple.object_type not in valid_types:
                    triple.object_type = self.config.entity_type_fallback
                triples.append(triple)
            result[chunk.chunk_id] = triples
        return result

    def evaluate_candidates(
        self,
        candidates: Sequence[CandidateTriple],
        *,
        batch_size: int | None = None,
    ) -> dict[int, TripleEvaluation]:
        """Run the evaluator model over extracted candidate triples."""
        if not candidates:
            return {}

        decisions: dict[int, TripleEvaluation] = {}
        effective_batch_size = max(1, int(batch_size or self.config.extraction_judge_batch_size))
        for start in range(0, len(candidates), effective_batch_size):
            judge_batch = candidates[start : start + effective_batch_size]
            response = self.llm_client.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_evaluator_prompt(judge_batch)},
                ],
                **self._completion_options(self.config.judge_model.lower()),
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(clean_json_text(content))
            parsed = TripleEvaluationsResponse.model_validate(payload)
            for item in parsed.triples:
                decisions[item.triple_id] = item
        return decisions

    @staticmethod
    def _is_fatal_api_error(exc: Exception) -> bool:
        """Return True for errors that will never succeed on retry."""
        msg = str(exc).lower()
        return any(
            code in msg
            for code in (
                "model_not_found",
                "invalid_api_key",
                "authentication",
                "unsupported_parameter",
                "unsupported_value",
                "401",
                "404",
            )
        )

    def process_batch_task(
        self, batch: Sequence[ChunkRecord]
    ) -> tuple[list[tuple], list[tuple], list[tuple], int, dict[str, int], bool]:
        """Extract and evaluate triples for one batch with retries.

        The final bool is True when the batch completed successfully (including “no triples” after a
        valid pipeline run). False means all retries failed — those chunks remain pending.
        """
        first_id = batch[0].chunk_id if batch else "-"
        logger.info("Batch started: %d chunks (first_id=%s)", len(batch), first_id)
        last_error: Exception | None = None
        for attempt in range(self.config.extraction_max_retries):
            try:
                results = self.extract_batch(batch)
                run_id = str(uuid.uuid4())
                triple_rows: list[tuple] = []
                entity_rows: list[tuple] = []
                rejection_rows: list[tuple] = []
                rejection_counts: Counter[str] = Counter()
                raw_triples_seen = 0
                candidate_triples: list[CandidateTriple] = []

                next_triple_id = 0
                for record in batch:
                    triples = results.get(record.chunk_id, [])
                    raw_triples_seen += len(triples)
                    for triple in triples:
                        subject_norm = normalize_entity(triple.subject)[: self.config.max_entity_length]
                        object_norm = normalize_entity(triple.object)[: self.config.max_entity_length]
                        predicate_norm = normalize_predicate(triple.predicate)
                        if not subject_norm or not predicate_norm or not object_norm:
                            continue
                        rejection_reason = self._triple_rejection_reason(subject_norm, predicate_norm, object_norm)
                        if rejection_reason:
                            rejection_counts[rejection_reason] += 1
                            rejection_rows.append(
                                (
                                    str(uuid.uuid4()),
                                    record.text_id,
                                    record.chunk_id,
                                    triple.subject[: self.config.max_entity_length],
                                    subject_norm,
                                    triple.predicate[: self.config.max_entity_length],
                                    predicate_norm,
                                    triple.object[: self.config.max_entity_length],
                                    object_norm,
                                    "deterministic_guard",
                                    rejection_reason,
                                    self.config.extraction_model,
                                    run_id,
                                )
                            )
                            continue

                        candidate_triples.append(
                            CandidateTriple(
                                triple_id=next_triple_id,
                                text_id=record.text_id,
                                chunk_id=record.chunk_id,
                                chunk_text=record.content,
                                subject_str=triple.subject[: self.config.max_entity_length],
                                subject_norm=subject_norm,
                                subject_type=triple.subject_type,
                                predicate_str=triple.predicate[: self.config.max_entity_length],
                                predicate_norm=predicate_norm,
                                object_str=triple.object[: self.config.max_entity_length],
                                object_norm=object_norm,
                                object_type=triple.object_type,
                            )
                        )
                        next_triple_id += 1

                decisions = self.evaluate_candidates(candidate_triples)
                missing_candidates = [
                    candidate for candidate in candidate_triples if candidate.triple_id not in decisions
                ]
                if missing_candidates:
                    retry_decisions = self.evaluate_candidates(
                        missing_candidates,
                        batch_size=len(missing_candidates),
                    )
                    recovered = 0
                    for triple_id, decision in retry_decisions.items():
                        if triple_id not in decisions:
                            decisions[triple_id] = decision
                            recovered += 1
                    if recovered:
                        logger.info(
                            "Evaluator retry recovered %d/%d missing decisions for batch",
                            recovered,
                            len(missing_candidates),
                        )
                    remaining_missing = [
                        candidate for candidate in missing_candidates if candidate.triple_id not in decisions
                    ]
                    if remaining_missing:
                        logger.warning(
                            "Evaluator retry still missing %d/%d decisions for batch",
                            len(remaining_missing),
                            len(missing_candidates),
                        )
                for candidate in candidate_triples:
                    decision = decisions.get(candidate.triple_id)
                    if decision is None:
                        rejection_counts["judge_missing_decision_after_retry"] += 1
                        rejection_rows.append(
                            (
                                str(uuid.uuid4()),
                                candidate.text_id,
                                candidate.chunk_id,
                                candidate.subject_str,
                                candidate.subject_norm,
                                candidate.predicate_str,
                                candidate.predicate_norm,
                                candidate.object_str,
                                candidate.object_norm,
                                "llm_judge",
                                "missing evaluator decision after retry",
                                self.config.judge_model,
                                run_id,
                            )
                        )
                        continue
                    if not decision.keep:
                        rejection_counts["judge_rejected"] += 1
                        rejection_rows.append(
                            (
                                str(uuid.uuid4()),
                                candidate.text_id,
                                candidate.chunk_id,
                                candidate.subject_str,
                                candidate.subject_norm,
                                candidate.predicate_str,
                                candidate.predicate_norm,
                                candidate.object_str,
                                candidate.object_norm,
                                "llm_judge",
                                (decision.reason or "rejected by evaluator")[: self.config.max_entity_length],
                                self.config.judge_model,
                                run_id,
                            )
                        )
                        continue

                    triple_rows.append(
                        (
                            str(uuid.uuid4()),
                            candidate.text_id,
                            candidate.chunk_id,
                            candidate.subject_str,
                            candidate.subject_norm,
                            candidate.subject_type,
                            candidate.predicate_str,
                            candidate.predicate_norm,
                            candidate.object_str,
                            candidate.object_norm,
                            candidate.object_type,
                            self.config.extraction_model,
                            run_id,
                        )
                    )
                    entity_rows.append(
                        (candidate.subject_norm, candidate.subject_type, candidate.chunk_id, candidate.text_id, "subject")
                    )
                    entity_rows.append(
                        (candidate.object_norm, candidate.object_type, candidate.chunk_id, candidate.text_id, "object")
                    )
                return triple_rows, entity_rows, rejection_rows, raw_triples_seen, dict(rejection_counts), True
            except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
                last_error = exc
                if self._is_fatal_api_error(exc):
                    logger.error("Fatal API error (no retry): %s", exc)
                    break
                if attempt < self.config.extraction_max_retries - 1:
                    time.sleep(self.config.extraction_retry_delay_seconds)
        if last_error is not None:
            logger.error("Extraction/evaluation batch failed after retries: %s", last_error)
            logger.warning(
                "Batch chunks remain pending (not written). For Ollama: use extraction_max_workers=1–2, "
                "smaller extraction_batch_size (2–3), and extraction_http_timeout_seconds=1800+ in config."
            )
        return [], [], [], 0, {}, False

    def insert_triples(self, triple_rows: list[tuple], entity_rows: list[tuple]) -> None:
        """Insert extracted triples and provenance rows."""
        con = self.storage.con
        if triple_rows:
            con.executemany(
                """
                INSERT INTO normalized_triples (
                    triple_id, text_id, chunk_id, subject_str, subject_norm, subject_type,
                    predicate_str, predicate_norm, object_str, object_norm, object_type,
                    model_name, run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                triple_rows,
            )
        if entity_rows:
            con.executemany(
                """
                INSERT INTO entity_chunks (
                    entity_norm, entity_type, chunk_id, text_id, role
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (entity_norm, chunk_id, role) DO UPDATE SET
                    entity_type = EXCLUDED.entity_type,
                    text_id = EXCLUDED.text_id
                """,
                entity_rows,
            )

    def insert_rejections(self, rejection_rows: list[tuple]) -> None:
        """Persist rejected extraction candidates for later inspection."""
        if not rejection_rows:
            return
        self.storage.con.executemany(
            """
            INSERT INTO extraction_rejections (
                rejection_id, text_id, chunk_id, subject_str, subject_norm,
                predicate_str, predicate_norm, object_str, object_norm,
                stage, reason, model_name, run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rejection_rows,
        )

    def refresh_entity_types(self) -> None:
        """Rebuild aggregate entity types from entity_chunks."""
        con = self.storage.con
        con.execute("DELETE FROM entity_types")
        con.execute(
            """
            INSERT INTO entity_types (entity_norm, entity_type, subtype, mention_count)
            SELECT entity_norm, entity_type, NULL AS subtype, COUNT(*) as mention_count
            FROM entity_chunks
            GROUP BY 1, 2
            """
        )

    def _preflight_model_check(self, model_name: str, *, role: str) -> None:
        """Run a minimal API call to verify model access before bulk extraction."""
        logger.info(
            "Preflight check (%s): model=%s base=%s",
            role,
            model_name,
            self.config.extraction_api_base,
        )
        try:
            self.llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_completion_tokens=8,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_fatal_api_error(exc):
                raise RuntimeError(
                    f"Preflight failed — model '{model_name}' is not accessible. "
                    f"Override the model in config or CLI. Error: {exc}"
                ) from exc
            logger.warning("Preflight non-fatal warning: %s", exc)

    def _preflight_check(self) -> None:
        """Run minimal API calls to verify extraction and judge model access."""
        self._preflight_model_check(self.config.extraction_model, role="extraction")
        if self.config.judge_model != self.config.extraction_model:
            self._preflight_model_check(self.config.judge_model, role="judge")
        else:
            logger.info("Preflight: judge_model same as extraction_model — single model check only")

    def extract_pending(
        self,
        limit: int | None = None,
        levels: Sequence[str] = ("local",),
    ) -> dict[str, int]:
        """Extract triples for all pending chunks."""
        self.ensure_schema()
        total_pending_before_run = self.count_pending_chunks(levels=levels)
        chunks = self.fetch_pending_chunks(limit=limit, levels=levels)
        if not chunks:
            return {
                "pending_chunks": 0,
                "selected_pending_chunks": 0,
                "total_pending_before_run": total_pending_before_run,
                "remaining_pending_chunks": 0,
                "triples_inserted": 0,
                "raw_triples_seen": 0,
                "rejected_triples": 0,
                "skipped_chunks": 0,
                "entities_tracked": self.storage.con.execute("SELECT COUNT(*) FROM entity_chunks").fetchone()[0],
            }

        selected_pending_chunks = len(chunks)
        skip_rows: list[tuple[str, str]] = []
        skipped_chunk_reasons: Counter[str] = Counter()
        extractable_chunks: list[ChunkRecord] = []
        for chunk in chunks:
            reason = self._chunk_skip_reason(chunk.content)
            if reason:
                skip_rows.append((chunk.chunk_id, reason))
                skipped_chunk_reasons[reason] += 1
            else:
                extractable_chunks.append(chunk)

        if skip_rows:
            self.insert_chunk_skips(skip_rows)
            logger.info(
                "Skipped %d/%d selected chunks before extraction: %s",
                len(skip_rows),
                selected_pending_chunks,
                dict(skipped_chunk_reasons),
            )

        if not extractable_chunks:
            remaining_pending_chunks = self.count_pending_chunks(levels=levels)
            return {
                "pending_chunks": selected_pending_chunks,
                "selected_pending_chunks": selected_pending_chunks,
                "total_pending_before_run": total_pending_before_run,
                "remaining_pending_chunks": remaining_pending_chunks,
                "triples_inserted": 0,
                "raw_triples_seen": 0,
                "rejected_triples": 0,
                "skipped_chunks": len(skip_rows),
                "entity_mentions_inserted": 0,
                "entity_types_rows": self.storage.con.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0],
            }

        self._preflight_check()
        logger.info(
            "Extracting %d/%d selected pending chunks in %d-chunk batches …",
            len(extractable_chunks),
            selected_pending_chunks,
            self.config.extraction_batch_size,
        )
        batches = [
            extractable_chunks[i : i + self.config.extraction_batch_size]
            for i in range(0, len(extractable_chunks), self.config.extraction_batch_size)
        ]
        total_batches = len(batches)
        logger.info(
            "Queued %d batch jobs (extraction_max_workers=%d). "
            "Progress logs appear after each batch completes; the first batch often takes several minutes "
            "with local Ollama — reduce extraction_max_workers to 1–2 if requests queue or stall.",
            total_batches,
            self.config.extraction_max_workers,
        )
        progress_interval = max(1, min(25, total_batches // 20 or 1))
        total_triples = 0
        total_entities = 0
        raw_triples_seen = 0
        rejection_counts: Counter[str] = Counter()
        completed_batches = 0
        completed_chunks = 0
        failed_batches = 0
        started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.config.extraction_max_workers) as executor:
            futures = {executor.submit(self.process_batch_task, batch): batch for batch in batches}
            for future in as_completed(futures):
                triple_rows, entity_rows, rejection_rows, batch_raw_triples, batch_rejections, batch_ok = (
                    future.result()
                )
                batch = futures[future]
                self.insert_triples(triple_rows, entity_rows)
                self.insert_rejections(rejection_rows)
                total_triples += len(triple_rows)
                total_entities += len(entity_rows)
                raw_triples_seen += batch_raw_triples
                rejection_counts.update(batch_rejections)
                completed_batches += 1
                if batch_ok:
                    completed_chunks += len(batch)
                else:
                    failed_batches += 1

                should_log_progress = (
                    completed_batches == 1
                    or completed_batches == total_batches
                    or completed_batches % progress_interval == 0
                )
                if should_log_progress:
                    elapsed = max(0.001, time.perf_counter() - started_at)
                    chunks_per_second = completed_chunks / elapsed
                    triples_per_second = total_triples / elapsed if total_triples else 0.0
                    remaining_chunks = len(extractable_chunks) - completed_chunks
                    eta_seconds = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0.0
                    logger.info(
                        (
                            "Extraction progress: %d/%d chunks done (%.1f%%), %d/%d batches, "
                            "%d batch failures, %d triples, %.2f chunks/s, %.2f triples/s, elapsed=%s, eta=%s"
                        ),
                        completed_chunks,
                        len(extractable_chunks),
                        100.0 * completed_chunks / max(1, len(extractable_chunks)),
                        completed_batches,
                        total_batches,
                        failed_batches,
                        total_triples,
                        chunks_per_second,
                        triples_per_second,
                        self._format_seconds(elapsed),
                        self._format_seconds(eta_seconds),
                    )

        self.refresh_entity_types()
        elapsed = time.perf_counter() - started_at
        remaining_pending_chunks = self.count_pending_chunks(levels=levels)
        logger.info(
            "Extraction complete: %d extracted chunks (%d selected, %d skipped), %d triples, %d entity mentions in %s (%d remaining)",
            len(extractable_chunks),
            selected_pending_chunks,
            len(skip_rows),
            total_triples,
            total_entities,
            self._format_seconds(elapsed),
            remaining_pending_chunks,
        )
        if rejection_counts:
            logger.info(
                "Rejected %d triples by rule: %s",
                sum(rejection_counts.values()),
                dict(rejection_counts),
            )
        return {
            "pending_chunks": selected_pending_chunks,
            "selected_pending_chunks": selected_pending_chunks,
            "total_pending_before_run": total_pending_before_run,
            "remaining_pending_chunks": remaining_pending_chunks,
            "triples_inserted": total_triples,
            "raw_triples_seen": raw_triples_seen,
            "rejected_triples": sum(rejection_counts.values()),
            "skipped_chunks": len(skip_rows),
            "entity_mentions_inserted": total_entities,
            "entity_types_rows": self.storage.con.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0],
        }
