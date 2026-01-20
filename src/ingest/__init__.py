"""Data ingestion pipeline for PhilosophAI."""
from .cleaner import TripleCleaner
from .canonicalizer import EntityCanonicalizer
from .embedder import ChunkEmbedder

__all__ = [
    "TripleCleaner",
    "EntityCanonicalizer", 
    "ChunkEmbedder",
]
