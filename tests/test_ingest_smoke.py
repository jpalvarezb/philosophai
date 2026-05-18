"""End-to-end smoke test for the multi-agent ingestion pipeline.

Exercises Load -> Audit -> Chunk -> Extract -> Clean with mocked LLM
responses and verifies database state after each step.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.agents import CleaningAgent, CorpusAuditAgent
from src.ingest.chunker import CorpusChunker
from src.ingest.cleaner import TripleCleaner
from src.ingest.config import IngestConfig
from src.ingest.extractor import (
    _EVALUATOR_SYSTEM_PROMPT,
    ChunkRecord,
    TripleExtractor,
    build_system_prompt,
)
from src.ingest.loader import CorpusLoader
from src.storage import DuckDBStorage


SAMPLE_TEXTS = {
    "0001_Republic_Plato.txt": (
        "Justice is the excellence of the soul.\n\n"
        "The myth of the cave illustrates the journey from ignorance to knowledge.\n\n"
        "Socrates argues that the philosopher-king is the ideal ruler.\n\n"
        "The tripartite soul consists of reason, spirit, and appetite."
    ),
    "0002_Nicomachean_Ethics_Aristotle.txt": (
        "Happiness is the highest good achievable by action.\n\n"
        "Virtue is a mean between two extremes.\n\n"
        "The good life requires both moral virtue and intellectual virtue.\n\n"
        "Friendship is essential to the flourishing life."
    ),
}


def _make_mock_response(content: dict[str, Any]) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = json.dumps(content)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _build_mock_llm_client(extraction_config: IngestConfig) -> MagicMock:
    """Build a mock OpenAI client whose .chat.completions.create returns
    different responses depending on the system prompt content."""

    audit_response = _make_mock_response({
        "chunk_max_chars": 1800,
        "chunk_overlap": 300,
        "chunk_method": "paragraph",
        "boilerplate_patterns": [r"^Page\s+\d+$"],
        "discovered_subtypes": {"Person": ["Ancient Greek Philosopher"]},
        "findings": ["Corpus consists of classical philosophical texts."],
    })

    cleaning_response = _make_mock_response({
        "entity_patterns": ["%gutenberg%", "%ebook%"],
        "predicate_patterns": ["%appears on page%"],
        "type_rules": {},
        "explanations": {
            "%gutenberg%": "Project Gutenberg metadata",
            "%ebook%": "eBook boilerplate",
        },
    })

    validation_keep = _make_mock_response({"keep": True, "rationale": "Clearly corpus metadata noise."})

    def _extraction_response_for(chunks: list[dict]) -> MagicMock:
        """Build a plausible extraction response for given chunk IDs."""
        result_chunks = []
        for chunk_info in chunks:
            result_chunks.append({
                "chunk_id": chunk_info["id"],
                "triples": [
                    {
                        "subject": "Plato",
                        "subject_type": "Person",
                        "predicate": "argues_for",
                        "object": "justice",
                        "object_type": "Concept",
                    },
                    {
                        "subject": "virtue",
                        "subject_type": "Concept",
                        "predicate": "is_excellence_of",
                        "object": "soul",
                        "object_type": "Concept",
                    },
                ],
            })
        return _make_mock_response({"chunks": result_chunks})

    def _side_effect(**kwargs):
        messages = kwargs.get("messages", [])
        system_content = messages[0]["content"] if messages else ""

        if "corpus-profiling agent" in system_content:
            return audit_response
        if "cleaning-rules agent" in system_content:
            return cleaning_response
        if "reviewing a candidate SQL LIKE" in system_content:
            return validation_keep
        if "triple quality evaluator" in system_content:
            user_content = messages[1]["content"] if len(messages) > 1 else ""
            triple_ids = [int(match) for match in re.findall(r"Triple ID:\s*(\d+)", user_content)]
            return _make_mock_response({
                "triples": [
                    {
                        "triple_id": triple_id,
                        "keep": True,
                        "reason": "Grounded semantic content.",
                    }
                    for triple_id in triple_ids
                ]
            })
        if "information extraction engine" in system_content:
            user_content = messages[1]["content"] if len(messages) > 1 else ""
            chunk_ids = []
            for line in user_content.split("\n"):
                if line.startswith("ID: "):
                    chunk_ids.append({"id": line[4:].strip()})
            if not chunk_ids:
                chunk_ids = [{"id": "unknown"}]
            return _extraction_response_for(chunk_ids)

        return _make_mock_response({})

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=_side_effect)
    return client


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a small test corpus in a temp directory."""
    resources = tmp_path / "resources"
    resources.mkdir()
    for filename, content in SAMPLE_TEXTS.items():
        (resources / filename).write_text(content, encoding="utf-8")
    return resources


@pytest.fixture
def db_storage(tmp_path: Path) -> DuckDBStorage:
    return DuckDBStorage(tmp_path / "smoke.duckdb")


def _table_count(storage: DuckDBStorage, table: str) -> int:
    return storage.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _table_exists(storage: DuckDBStorage, table: str) -> bool:
    tables = [row[0] for row in storage.con.execute("SHOW TABLES").fetchall()]
    return table in tables


def test_build_system_prompt_uses_config_types():
    """The extraction system prompt should reflect config entity types."""
    config = IngestConfig()
    prompt = build_system_prompt(config)
    for type_name in config.entity_types:
        assert type_name in prompt
    assert config.entity_type_fallback in prompt


def test_extractor_prompt_mentions_examples_and_guardrails():
    """Extraction prompt should include few-shot guidance and anti-hallucination rules."""
    prompt = build_system_prompt(IngestConfig()).lower()
    assert "example 1 (keep semantic claims)" in prompt
    assert "example 2 (reject structural/editorial references)" in prompt
    assert "example 3 (reject unresolved pronoun speaker)" in prompt
    assert "example 4 (reject vague \"is\" relation)" in prompt
    assert "skip speculative or hedged claims" in prompt
    assert "unresolved pronoun/speaker reference" in prompt
    assert "chapter/section/verse/stanza/folio numbering" in prompt


def test_extractor_fatal_error_classifier_is_specific():
    """Generic 400s should not be treated as model-inaccessible fatal errors."""

    class DummyExc(Exception):
        pass

    assert TripleExtractor._is_fatal_api_error(DummyExc("model_not_found"))
    assert TripleExtractor._is_fatal_api_error(DummyExc("unsupported_parameter"))
    assert not TripleExtractor._is_fatal_api_error(
        DummyExc("Error code: 400 - output limit was reached. Please try again.")
    )


def test_extractor_rule_filters():
    """Extractor should reject obvious low-quality triples."""
    assert TripleExtractor._triple_rejection_reason("you", "may demand refund if", "copy") == "pronoun_subject"
    assert TripleExtractor._triple_rejection_reason("chapter", "refers to", "concept") == "structural_subject"
    assert TripleExtractor._triple_rejection_reason("N.", "is", "triumphant") == "placeholder_subject"
    assert TripleExtractor._triple_rejection_reason("Horus", "guides", "N.") == "placeholder_object"
    assert TripleExtractor._triple_rejection_reason("A", "is", "concept") == "single_letter_subject"
    assert (
        TripleExtractor._triple_rejection_reason("project gutenberg", "is", "license")
        == "legal_boilerplate_triple"
    )
    assert (
        TripleExtractor._triple_rejection_reason(
            "Every one who exalts himself will be humbled and he who humbles himself",
            "teaches",
            "humility",
        )
        == "entity_too_long_subject"
    )
    assert (
        TripleExtractor._triple_rejection_reason(
            "Plato",
            "argues",
            "that which he has led upon earth in the life of a monk",
        )
        == "entity_too_long_object"
    )
    assert TripleExtractor._triple_rejection_reason("Velleity", "is", "desire") is None
    assert TripleExtractor._triple_rejection_reason("Egyptians living under the eleventh dynasty", "believed in", "afterlife") is None


def test_evaluator_prompt_mentions_structural_and_editorial_rejections():
    """Evaluator prompt should explicitly reject structural/editorial survivor patterns."""
    prompt = _EVALUATOR_SYSTEM_PROMPT.lower()
    assert "chapter xii" in prompt
    assert "publication metadata" in prompt
    assert "translators" in prompt
    assert "is_known_in" in prompt
    assert "is_opened_in" in prompt
    assert "the speaker" in prompt
    assert "the narrator" in prompt


def test_extractor_evaluator_rejects_bad_candidate(db_storage: DuckDBStorage):
    """Evaluator should veto extracted triples that fail semantic quality requirements."""
    config = IngestConfig()
    extraction_response = _make_mock_response({
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "triples": [
                    {
                        "subject": "Plato",
                        "subject_type": "Person",
                        "predicate": "argues_for",
                        "object": "justice",
                        "object_type": "Concept",
                    },
                    {
                        "subject": "Chapter XII",
                        "subject_type": "Work",
                        "predicate": "describes",
                        "object": "afterlife",
                        "object_type": "Concept",
                    },
                    {
                        "subject": "Papyrus of Ani",
                        "subject_type": "Work",
                        "predicate": "contains_title",
                        "object": "Chapter of Coming Forth by Day",
                        "object_type": "Work",
                    },
                ],
            }
        ]
    })
    judge_response = _make_mock_response({
        "triples": [
            {"triple_id": 0, "keep": True, "reason": "Grounded philosophical claim."},
            {"triple_id": 1, "keep": False, "reason": "Structural chapter reference, not content."},
            {"triple_id": 2, "keep": False, "reason": "Editorial title-listing fact, not semantic content."},
        ]
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(side_effect=[extraction_response, judge_response])

    extractor = TripleExtractor(db_storage, llm_client=mock_client, config=config)
    batch = [
        ChunkRecord(
            chunk_id="chunk-1",
            text_id=1,
            content=(
                "Plato argues for justice in the ideal city. "
                "Chapter XII names the later section of the book. "
                "The Papyrus of Ani contains the title 'Chapter of Coming Forth by Day.'"
            ),
        )
    ]

    triple_rows, entity_rows, rejection_rows, raw_triples_seen, rejection_counts, batch_ok = (
        extractor.process_batch_task(batch)
    )

    assert batch_ok is True
    assert raw_triples_seen == 3
    assert len(triple_rows) == 1
    assert len(entity_rows) == 2
    assert len(rejection_rows) == 2
    assert all(row[9] == "llm_judge" for row in rejection_rows)
    assert "Structural chapter reference" in rejection_rows[0][10]
    assert "Editorial title-listing fact" in rejection_rows[1][10]
    assert rejection_counts["judge_rejected"] == 2


def test_extractor_retries_missing_judge_decisions_once(db_storage: DuckDBStorage):
    """Missing evaluator decisions should be retried once before acceptance/rejection."""
    config = IngestConfig()
    extraction_response = _make_mock_response({
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "triples": [
                    {
                        "subject": "Plato",
                        "subject_type": "Person",
                        "predicate": "argues_for",
                        "object": "justice",
                        "object_type": "Concept",
                    },
                    {
                        "subject": "Aristotle",
                        "subject_type": "Person",
                        "predicate": "defines",
                        "object": "virtue",
                        "object_type": "Concept",
                    },
                ],
            }
        ]
    })
    first_judge_response = _make_mock_response({
        "triples": [
            {"triple_id": 0, "keep": True, "reason": "Grounded semantic claim."},
        ]
    })
    retry_judge_response = _make_mock_response({
        "triples": [
            {"triple_id": 1, "keep": True, "reason": "Grounded semantic claim on retry."},
        ]
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(
        side_effect=[extraction_response, first_judge_response, retry_judge_response]
    )

    extractor = TripleExtractor(db_storage, llm_client=mock_client, config=config)
    batch = [
        ChunkRecord(
            chunk_id="chunk-1",
            text_id=1,
            content="Plato argues for justice. Aristotle defines virtue as a mean.",
        )
    ]

    triple_rows, entity_rows, rejection_rows, raw_triples_seen, rejection_counts, batch_ok = (
        extractor.process_batch_task(batch)
    )

    assert batch_ok is True
    assert raw_triples_seen == 2
    assert len(triple_rows) == 2
    assert len(entity_rows) == 4
    assert rejection_rows == []
    assert rejection_counts == {}
    assert mock_client.chat.completions.create.call_count == 3


def test_extractor_fails_closed_after_missing_judge_retry(db_storage: DuckDBStorage):
    """A candidate still missing after one retry should be rejected explicitly."""
    config = IngestConfig()
    extraction_response = _make_mock_response({
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "triples": [
                    {
                        "subject": "Plato",
                        "subject_type": "Person",
                        "predicate": "argues_for",
                        "object": "justice",
                        "object_type": "Concept",
                    },
                    {
                        "subject": "Aristotle",
                        "subject_type": "Person",
                        "predicate": "defines",
                        "object": "virtue",
                        "object_type": "Concept",
                    },
                ],
            }
        ]
    })
    first_judge_response = _make_mock_response({
        "triples": [
            {"triple_id": 0, "keep": True, "reason": "Grounded semantic claim."},
        ]
    })
    retry_judge_response = _make_mock_response({"triples": []})
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(
        side_effect=[extraction_response, first_judge_response, retry_judge_response]
    )

    extractor = TripleExtractor(db_storage, llm_client=mock_client, config=config)
    batch = [
        ChunkRecord(
            chunk_id="chunk-1",
            text_id=1,
            content="Plato argues for justice. Aristotle defines virtue as a mean.",
        )
    ]

    triple_rows, entity_rows, rejection_rows, raw_triples_seen, rejection_counts, batch_ok = (
        extractor.process_batch_task(batch)
    )

    assert batch_ok is True
    assert raw_triples_seen == 2
    assert len(triple_rows) == 1
    assert len(entity_rows) == 2
    assert len(rejection_rows) == 1
    assert rejection_rows[0][9] == "llm_judge"
    assert rejection_rows[0][10] == "missing evaluator decision after retry"
    assert rejection_counts["judge_missing_decision_after_retry"] == 1
    assert mock_client.chat.completions.create.call_count == 3


def test_extractor_chunk_skip_rules():
    """Chunk prefilter should catch obvious structural/legal chunks."""
    toc_chunk = "Table of Contents\nChapter I\nChapter II\nChapter III"
    legal_chunk = "Project Gutenberg License\nAll rights reserved. Terms of use apply."
    normal_chunk = "Velleity is the lowest degree of desire in this account."
    assert TripleExtractor._chunk_skip_reason(toc_chunk) in {"table_of_contents_chunk", "structural_heading_chunk"}
    assert TripleExtractor._chunk_skip_reason(legal_chunk) == "legal_boilerplate_chunk"
    assert TripleExtractor._chunk_skip_reason(normal_chunk) is None

    toc_listing = "\n".join(
        [f"Chapter {n} — Title {n}" for n in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]]
        + ["Some filler line"]
    )
    assert TripleExtractor._chunk_skip_reason(toc_listing) in {
        "toc_like_chapter_listing",
        "structural_heading_chunk",
    }

    toc_mixed = "\n".join(
        [f"Chapter {n} ..... p. {i*10}" for i, n in enumerate(["I", "II", "III", "IV", "V", "VI"], 1)]
        + ["Some other content here", "More content here", "Even more here"]
    )
    assert TripleExtractor._chunk_skip_reason(toc_mixed) == "toc_like_chapter_listing"

    index_chunk = "\n".join([
        "Afterlife, see also Underworld",
        "Anubis, see also Jackal-headed god",
        "Ba, see also Soul",
        "Cosmogony, see under Creation myths",
        "Some normal line here",
    ])
    assert TripleExtractor._chunk_skip_reason(index_chunk) == "index_glossary_chunk"

    short_header = "Chapter XII — The Weighing of the Heart"
    assert TripleExtractor._chunk_skip_reason(short_header) == "short_header_only_chunk"

    real_content = (
        "Plato argues that the soul is immortal and that true knowledge "
        "comes from recollection of the Forms. The philosopher must turn "
        "away from the shadows on the cave wall toward the light of the Good."
    )
    assert TripleExtractor._chunk_skip_reason(real_content) is None


def test_corpus_audit_subtype_validator_rejects_instances():
    """Subtype labels should be generic categories, not entity instances."""
    assert CorpusAuditAgent._is_category_label("Ancient Greek Philosopher")
    assert CorpusAuditAgent._is_category_label("Philosophical Text")
    assert not CorpusAuditAgent._is_category_label("Plato")
    assert not CorpusAuditAgent._is_category_label("Republic")
    assert not CorpusAuditAgent._is_category_label("Project Gutenberg")


def test_corpus_audit_filters_instance_like_subtypes(db_storage: DuckDBStorage, corpus_dir: Path):
    """Audit output should drop entity instances that masquerade as subtypes."""
    config = IngestConfig(resources_dir=str(corpus_dir))

    loader = CorpusLoader(db_storage, config=config)
    loader.load_resources()

    response = _make_mock_response(
        {
            "chunk_max_chars": 1800,
            "chunk_overlap": 300,
            "chunk_method": "paragraph",
            "boilerplate_patterns": [],
            "discovered_subtypes": {
                "Person": ["Plato", "Ancient Greek Philosopher"],
                "Work": ["Republic", "Philosophical Text"],
                "Organization": ["Project Gutenberg"],
            },
            "findings": [],
        }
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=response)

    agent = CorpusAuditAgent(db_storage, llm_client=mock_client, config=config)
    _, report = agent.run()

    assert report.discovered_subtypes == {
        "Person": ["Ancient Greek Philosopher"],
        "Work": ["Philosophical Text"],
    }


def test_corpus_audit_clamps_bad_values(db_storage: DuckDBStorage, corpus_dir: Path):
    """Values outside sane ranges get clamped."""
    config = IngestConfig(resources_dir=str(corpus_dir))

    loader = CorpusLoader(db_storage, config=config)
    loader.load_resources()

    bad_response = _make_mock_response({
        "chunk_max_chars": 999999,
        "chunk_overlap": -50,
        "chunk_method": "invalid_method",
        "boilerplate_patterns": ["(invalid[regex"],
        "discovered_subtypes": {"FakeType": ["sub"]},
        "findings": [],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=bad_response)

    agent = CorpusAuditAgent(db_storage, llm_client=mock_client, config=config)
    updated_config, report = agent.run()

    assert updated_config.chunk_max_chars <= 5000
    assert updated_config.chunk_overlap >= 0
    assert updated_config.chunk_method in ("paragraph", "semantic")
    assert report.boilerplate_patterns == []
    assert "FakeType" not in report.discovered_subtypes


def test_cleaning_agent_rejects_invalid_patterns(db_storage: DuckDBStorage):
    """SQL LIKE patterns with injection-like content get filtered out."""
    agent = CleaningAgent(db_storage, llm_client=None, config=IngestConfig())
    assert not agent._is_valid_like_pattern("'; DROP TABLE--")
    assert not agent._is_valid_like_pattern("")
    assert agent._is_valid_like_pattern("%gutenberg%")
    assert agent._is_valid_like_pattern("%chapter %")


def test_full_pipeline_smoke(db_storage: DuckDBStorage, corpus_dir: Path):
    """Run Load -> Audit -> Chunk -> Extract -> Clean and verify DB state."""
    config = IngestConfig(resources_dir=str(corpus_dir))
    mock_client = _build_mock_llm_client(config)

    # Step 0: Load
    loader = CorpusLoader(db_storage, config=config)
    load_result = loader.load_resources()
    assert load_result["loaded"] == 2
    assert _table_count(db_storage, "files") == 2
    assert _table_count(db_storage, "raw_texts") == 2

    # Step 1: Corpus Audit
    audit_agent = CorpusAuditAgent(db_storage, llm_client=mock_client, config=config)
    config, report = audit_agent.run()
    assert report.chunk_recommendations["chunk_max_chars"] == 1800
    assert len(report.boilerplate_patterns) >= 1

    # Step 2: Chunk
    chunker = CorpusChunker(db_storage, config=config)
    chunk_result = chunker.chunk_all()
    assert chunk_result["chunks_inserted"] > 0
    assert _table_exists(db_storage, "chunks")
    chunk_count = _table_count(db_storage, "chunks")
    assert chunk_count >= 2

    # Step 3: Extract (with mocked LLM)
    extractor = TripleExtractor(db_storage, llm_client=mock_client, config=config)
    extract_result = extractor.extract_pending()
    assert extract_result["selected_pending_chunks"] > 0
    assert extract_result["total_pending_before_run"] >= extract_result["selected_pending_chunks"]
    assert extract_result["remaining_pending_chunks"] == 0
    assert extract_result["raw_triples_seen"] >= extract_result["triples_inserted"]
    assert extract_result["rejected_triples"] >= 0
    assert extract_result["skipped_chunks"] >= 0
    assert extract_result["triples_inserted"] > 0
    assert _table_exists(db_storage, "normalized_triples")
    assert _table_exists(db_storage, "entity_chunks")
    assert _table_exists(db_storage, "entity_types")
    assert _table_exists(db_storage, "extraction_rejections")
    triple_count = _table_count(db_storage, "normalized_triples")
    assert triple_count > 0

    # Verify typed extraction
    types = db_storage.con.execute(
        "SELECT DISTINCT subject_type FROM normalized_triples"
    ).fetchall()
    type_names = {row[0] for row in types}
    assert type_names <= set(config.entity_types), f"Unexpected types: {type_names}"

    # Step 4: Dynamic Clean
    cleaning_agent = CleaningAgent(db_storage, llm_client=mock_client, config=config)
    config, rules = cleaning_agent.run()
    assert isinstance(rules.entity_patterns, list)
    assert isinstance(rules.predicate_patterns, list)
    assert len(rules.predicate_patterns) > 0

    cleaner = TripleCleaner(db_storage, config=config)
    clean_result = cleaner.clean(dry_run=False)
    assert clean_result["total"] > 0
    assert clean_result["clean"] > 0
    assert _table_exists(db_storage, "normalized_triples_clean")

    # Idempotency check: re-loading should skip existing
    reload_result = loader.load_resources(skip_existing=True)
    assert reload_result["loaded"] == 0
    assert reload_result["skipped"] == 2

    # Re-extraction should find no pending chunks
    re_extract = extractor.extract_pending()
    assert re_extract["pending_chunks"] == 0
    assert re_extract["remaining_pending_chunks"] == 0


def test_pipeline_empty_corpus(db_storage: DuckDBStorage, tmp_path: Path):
    """Pipeline handles an empty corpus directory gracefully."""
    empty_dir = tmp_path / "empty_resources"
    empty_dir.mkdir()
    config = IngestConfig(resources_dir=str(empty_dir))

    loader = CorpusLoader(db_storage, config=config)
    result = loader.load_resources()
    assert result["loaded"] == 0

    audit = CorpusAuditAgent(db_storage, llm_client=None, config=config)
    updated_config, report = audit.run()
    assert report.raw_profiles.get("document_count", 0) == 0
    assert updated_config.chunk_max_chars == config.chunk_max_chars

    chunker = CorpusChunker(db_storage, config=config)
    chunk_result = chunker.chunk_all()
    assert chunk_result["chunks_inserted"] == 0

    cleaning_agent = CleaningAgent(db_storage, llm_client=None, config=config)
    updated_config, rules = cleaning_agent.run()
    assert rules.entity_patterns == []
