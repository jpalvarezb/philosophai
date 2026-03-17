"""Data ingestion pipeline for PhilosophAI."""
from .config import CleaningRules, CorpusAuditReport, IngestConfig
from .cleaner import TripleCleaner
from .canonicalizer import EntityCanonicalizer
from .embedder import ChunkEmbedder
from .extractor import TripleExtractor
from .chunker import CorpusChunker
from .loader import CorpusLoader

__all__ = [
    "CleaningRules",
    "CorpusAuditReport",
    "IngestConfig",
    "TripleCleaner",
    "EntityCanonicalizer",
    "ChunkEmbedder",
    "TripleExtractor",
    "CorpusChunker",
    "CorpusLoader",
]
