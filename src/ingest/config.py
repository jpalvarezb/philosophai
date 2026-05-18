"""Shared configuration objects for the ingestion pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ENTITY_TYPES: dict[str, str] = {
    "Person": "Individual humans (Plato, Kant, Einstein)",
    "Organization": "Named institutions, formal bodies, or established schools of thought (the Stoics, Vienna Circle, UNESCO, the Catholic Church). Do NOT use for generic groups of people (monks, brethren, clerics) — those are Person.",
    "Concept": "Abstract ideas, theories, principles, arguments (virtue, categorical imperative, entropy)",
    "Work": "Named documents, texts, publications (Republic, GDPR, Critique of Pure Reason)",
    "Event": "Named happenings, periods, movements (Enlightenment, French Revolution)",
    "Place": "Geographic locations, regions (Athens, Prussia, Silicon Valley)",
    "Method": "Processes, techniques, practices (dialectics, phenomenological reduction)",
    "Technology": "Tools, systems, platforms (rare in philosophy, critical in technical domains)",
}


@dataclass(slots=True)
class CleaningRules:
    """Dynamic cleaning rules proposed by the cleaning agent."""

    entity_patterns: list[str] = field(default_factory=list)
    predicate_patterns: list[str] = field(default_factory=list)
    provenance_failed_entities: list[str] = field(default_factory=list)
    type_rules: dict[str, list[str]] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CleaningRules":
        if not data:
            return cls()
        return cls(
            entity_patterns=list(data.get("entity_patterns", [])),
            predicate_patterns=list(data.get("predicate_patterns", [])),
            provenance_failed_entities=list(data.get("provenance_failed_entities", [])),
            type_rules={k: list(v) for k, v in dict(data.get("type_rules", {})).items()},
            explanations=dict(data.get("explanations", {})),
        )


@dataclass(slots=True)
class CorpusAuditReport:
    """Persisted summary of corpus profiling and agent decisions."""

    chunk_recommendations: dict[str, Any] = field(default_factory=dict)
    boilerplate_patterns: list[str] = field(default_factory=list)
    discovered_subtypes: dict[str, list[str]] = field(default_factory=dict)
    sample_findings: list[str] = field(default_factory=list)
    raw_profiles: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CorpusAuditReport":
        if not data:
            return cls()
        return cls(
            chunk_recommendations=dict(data.get("chunk_recommendations", {})),
            boilerplate_patterns=list(data.get("boilerplate_patterns", [])),
            discovered_subtypes={
                key: list(values) for key, values in dict(data.get("discovered_subtypes", {})).items()
            },
            sample_findings=list(data.get("sample_findings", [])),
            raw_profiles=dict(data.get("raw_profiles", {})),
        )


@dataclass(slots=True)
class IngestConfig:
    """Centralized ingestion settings shared across pipeline stages."""

    resources_dir: str = "data/resources"
    supported_formats: list[str] = field(default_factory=lambda: [".txt", ".epub", ".pdf"])

    chunk_max_chars: int = 1800
    chunk_overlap: int = 300
    chunk_method: str = "paragraph"

    extraction_model: str = "mistral:7b"
    extraction_api_base: str = "http://127.0.0.1:11434/v1"
    extraction_api_key_env: str = "OPENAI_API_KEY"
    extraction_batch_size: int = 5
    extraction_max_workers: int = 6
    extraction_max_retries: int = 3
    extraction_retry_delay_seconds: float = 2.0
    extraction_judge_batch_size: int = 15
    #: Per-request HTTP timeout for the OpenAI-compatible client (seconds). Use a large value (e.g.
    #: 1800) for local Ollama + big JSON payloads; None = library default (often too low for 5-chunk batches).
    extraction_http_timeout_seconds: float | None = None

    entity_types: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENTITY_TYPES))
    entity_type_fallback: str = "Concept"
    discovered_subtypes: dict[str, list[str]] = field(default_factory=dict)

    ocr_enabled: bool = False
    ocr_language: str = "eng"
    ocr_dpi: int = 200

    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100

    boilerplate_patterns: list[str] = field(default_factory=list)
    noise_entity_patterns: list[str] = field(default_factory=list)
    noise_predicate_patterns: list[str] = field(default_factory=list)
    provenance_threshold: float = 0.80
    min_entity_length: int = 2
    max_entity_length: int = 100

    predicate_threshold: float = 0.88
    entity_threshold: float = 0.92
    judge_model: str = "qwen3.5:9b"

    resolution: float = 0.8
    min_edge_weight: int = 1

    cleaning_rules: CleaningRules = field(default_factory=CleaningRules)
    audit_report: CorpusAuditReport = field(default_factory=CorpusAuditReport)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IngestConfig":
        """Hydrate config from a previously saved JSON payload."""
        if not data:
            return cls()

        payload = dict(data)
        payload["entity_types"] = dict(payload.get("entity_types", DEFAULT_ENTITY_TYPES))
        payload["discovered_subtypes"] = {
            key: list(values) for key, values in dict(payload.get("discovered_subtypes", {})).items()
        }
        payload["supported_formats"] = list(payload.get("supported_formats", [".txt", ".epub", ".pdf"]))
        payload["boilerplate_patterns"] = list(payload.get("boilerplate_patterns", []))
        payload["noise_entity_patterns"] = list(payload.get("noise_entity_patterns", []))
        payload["noise_predicate_patterns"] = list(payload.get("noise_predicate_patterns", []))
        payload["cleaning_rules"] = CleaningRules.from_dict(payload.get("cleaning_rules"))
        payload["audit_report"] = CorpusAuditReport.from_dict(payload.get("audit_report"))
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "IngestConfig":
        """Load config from disk."""
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        """Persist config to disk."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return config_path

    def merge_overrides(self, **overrides: Any) -> "IngestConfig":
        """Return a copy with selected keys overridden."""
        payload = self.to_dict()
        for key, value in overrides.items():
            if value is not None and key in payload:
                payload[key] = value
        return self.from_dict(payload)


def load_ingest_config(path: str | Path | None = None) -> IngestConfig:
    """Load a config from disk or return defaults."""
    if path is None:
        return IngestConfig()
    return IngestConfig.load(path)
