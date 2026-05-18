from __future__ import annotations

from pathlib import Path


from src.ingest.chunker import CorpusChunker
from src.ingest.cleaner import TripleCleaner
from src.ingest.config import CleaningRules, IngestConfig
from src.ingest.extractor import normalize_entity, normalize_predicate
from src.ingest.loader import parse_resource_metadata
from src.storage import DuckDBStorage


def test_ingest_config_round_trip(tmp_path: Path):
    config = IngestConfig(
        chunk_method="semantic",
        noise_entity_patterns=["%gutenberg%"],
    )
    config.cleaning_rules = CleaningRules(
        entity_patterns=["%chapter%"],
        provenance_failed_entities=["project gutenberg"],
    )

    path = tmp_path / "ingest_config.json"
    config.save(path)
    loaded = IngestConfig.load(path)

    assert loaded.chunk_method == "semantic"
    assert loaded.noise_entity_patterns == ["%gutenberg%"]
    assert loaded.cleaning_rules.entity_patterns == ["%chapter%"]
    assert loaded.cleaning_rules.provenance_failed_entities == ["project gutenberg"]


def test_parse_resource_metadata_uses_filename_convention(tmp_path: Path):
    path = tmp_path / "0001_Critique_of_Pure_Reason_Kant.txt"
    path.write_text("hello", encoding="utf-8")

    meta = parse_resource_metadata(path, fallback_id=99)

    assert meta.text_id == 2
    assert meta.title == "Critique of Pure Reason"
    assert meta.author_source == "Kant"
    assert meta.file_ext == ".txt"


def test_chunker_paragraph_mode_creates_document_chunk(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(db_path)
    chunker = CorpusChunker(
        storage,
        IngestConfig(chunk_max_chars=60, chunk_overlap=10, chunk_method="paragraph"),
    )

    text = (
        "Page 1\n\n"
        "First paragraph about justice and virtue.\n\n"
        "Second paragraph about reason and method.\n\n"
        "Third paragraph about the polis and citizenship."
    )
    payloads = chunker.chunk_text(text)

    assert any(payload.level == "local" for payload in payloads)
    assert any(payload.level == "document" for payload in payloads)
    assert all("Page 1" not in payload.content for payload in payloads)


def test_cleaner_builds_dynamic_noise_sql(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(db_path)
    cleaner = TripleCleaner(
        storage,
        IngestConfig(
            noise_entity_patterns=["%gutenberg%"],
            noise_predicate_patterns=["%copyright%"],
            cleaning_rules=CleaningRules(
                provenance_failed_entities=["chapter"],
                type_rules={"Work": ["%preface%"]},
            ),
        ),
    )

    sql = cleaner._build_noise_filter()

    assert "LOWER(subject_norm) LIKE '%gutenberg%'" in sql
    assert "LOWER(predicate_norm) LIKE '%copyright%'" in sql
    assert "LOWER(subject_norm) = 'chapter'" in sql
    assert "subject_type = 'Work'" in sql


def test_extractor_normalization_helpers():
    assert normalize_entity('  "Republic"  ') == "Republic"
    assert normalize_predicate("depends_on") == "depends on"
