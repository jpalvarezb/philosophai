"""Production chunking for raw corpus text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import IngestConfig

if TYPE_CHECKING:
    from ..storage import DuckDBStorage


logger = logging.getLogger(__name__)

PAGE_HEADER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ChunkPayload:
    """Intermediate chunk representation before DB insert."""

    level: str
    start_char: int
    end_char: int
    approx_tokens: int
    content: str


class CorpusChunker:
    """Build chunks from raw text stored in DuckDB."""

    def __init__(self, storage: "DuckDBStorage", config: IngestConfig | None = None):
        self.storage = storage
        self.config = config or IngestConfig()

    def ensure_schema(self) -> None:
        """Ensure the chunks table exists with the expected schema."""
        self.storage.con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                text_id INTEGER,
                position INTEGER,
                level VARCHAR,
                start_char INTEGER,
                end_char INTEGER,
                approx_tokens INTEGER,
                source_path VARCHAR,
                content VARCHAR
            )
            """)

    def chunk_all(self, skip_existing: bool = True) -> dict[str, int]:
        """Chunk all raw texts into the chunks table."""
        self.ensure_schema()
        con = self.storage.con

        if skip_existing:
            texts = con.execute("""
                SELECT f.text_id, f.file_path, r.content
                FROM files f
                INNER JOIN raw_texts r ON f.text_id = r.text_id
                WHERE f.text_id NOT IN (SELECT DISTINCT text_id FROM chunks)
                ORDER BY f.text_id
                """).fetchall()
        else:
            con.execute("DELETE FROM chunks")
            texts = con.execute("""
                SELECT f.text_id, f.file_path, r.content
                FROM files f
                INNER JOIN raw_texts r ON f.text_id = r.text_id
                ORDER BY f.text_id
                """).fetchall()

        inserted = 0
        total_texts = len(texts)
        for text_idx, (text_id, file_path, raw_text) in enumerate(texts, start=1):
            char_count = len(raw_text) if raw_text else 0
            logger.info(
                "[%d/%d] Chunking text_id=%s (%s chars) %s",
                text_idx,
                total_texts,
                text_id,
                f"{char_count:,}",
                Path(file_path).name if file_path else "",
            )
            payloads = self.chunk_text(raw_text)
            if not payloads:
                logger.warning(
                    "No chunks produced for text_id=%s path=%s", text_id, file_path
                )
                continue

            level_counts: dict[str, int] = {}
            sequence_position = 0
            for payload in payloads:
                level_counts[payload.level] = level_counts.get(payload.level, 0) + 1
                chunk_suffix = payload.level[0]
                if payload.level == "local":
                    sequence_position += 1
                    position = sequence_position
                else:
                    position = 0

                chunk_id = f"{text_id}_{chunk_suffix}{level_counts[payload.level]:04d}"
                con.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, text_id, position, level, start_char, end_char,
                        approx_tokens, source_path, content
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chunk_id,
                        text_id,
                        position,
                        payload.level,
                        payload.start_char,
                        payload.end_char,
                        payload.approx_tokens,
                        str(file_path),
                        payload.content,
                    ],
                )
                inserted += 1

        return {
            "texts_processed": len(texts),
            "chunks_inserted": inserted,
            "chunks_in_table": con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        }

    def chunk_text(self, text: str) -> list[ChunkPayload]:
        """Chunk one document according to the configured method."""
        cleaned = self._clean_text(text)
        if not cleaned:
            return []

        if self.config.chunk_method == "semantic":
            return self._semantic_chunk_text(cleaned)
        return self._paragraph_chunk_text(cleaned)

    def _clean_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text.replace("\r\n", "\n")
        text = re.sub(r"\t+", " ", text)
        patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.boilerplate_patterns
        ]

        lines: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if lines and lines[-1] == "":
                    continue
                lines.append("")
                continue

            if any(pattern.match(stripped) for pattern in PAGE_HEADER_PATTERNS):
                continue
            if patterns and any(pattern.search(stripped) for pattern in patterns):
                continue
            lines.append(stripped)

        collapsed = "\n".join(lines)
        collapsed = re.sub(r"[ ]{2,}", " ", collapsed)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return paragraphs or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _split_large_paragraphs(
        paragraphs: list[str], max_chars: int, overlap: int
    ) -> list[str]:
        result: list[str] = []
        for para in paragraphs:
            if len(para) <= max_chars:
                result.append(para)
                continue
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                segment = para[start:end].strip()
                if segment:
                    result.append(segment)
                if end == len(para):
                    break
                start = max(0, end - overlap)
        return result

    @staticmethod
    def _approx_tokens(chunk: str) -> int:
        return max(1, int(len(chunk) / 4))

    def _paragraph_chunk_text(self, cleaned: str) -> list[ChunkPayload]:
        max_chars = self.config.chunk_max_chars
        overlap = self.config.chunk_overlap
        paragraphs = self._split_into_paragraphs(cleaned)
        if not paragraphs:
            return []

        paragraphs = self._split_large_paragraphs(paragraphs, max_chars, overlap)
        canonical_text = "\n\n".join(paragraphs).strip()
        if not canonical_text:
            return []

        paragraph_offsets: list[int] = []
        cursor = 0
        for para in paragraphs:
            paragraph_offsets.append(cursor)
            cursor += len(para) + 2

        chunk_payloads: list[ChunkPayload] = []
        i = 0
        while i < len(paragraphs):
            start_idx = i
            current: list[str] = []
            length = 0

            while i < len(paragraphs):
                para = paragraphs[i]
                addition = len(para) if not current else len(para) + 2
                if current and length + addition > max_chars:
                    break
                current.append(para)
                length += addition
                i += 1

            if not current:
                current.append(paragraphs[i])
                i += 1

            end_idx = i - 1
            content = "\n\n".join(current).strip()
            if content:
                chunk_payloads.append(
                    ChunkPayload(
                        level="local",
                        start_char=paragraph_offsets[start_idx],
                        end_char=paragraph_offsets[end_idx] + len(paragraphs[end_idx]),
                        approx_tokens=self._approx_tokens(content),
                        content=content,
                    )
                )

            if i >= len(paragraphs):
                break

            overlap_chars = 0
            j = i - 1
            while j >= start_idx and overlap_chars < overlap:
                overlap_chars += len(paragraphs[j]) + 2
                j -= 1
            i = max(j + 1, start_idx + 1)

        if len(canonical_text) > max_chars or len(chunk_payloads) > 1:
            chunk_payloads.append(
                ChunkPayload(
                    level="document",
                    start_char=0,
                    end_char=len(canonical_text),
                    approx_tokens=self._approx_tokens(canonical_text),
                    content=canonical_text,
                )
            )
        return chunk_payloads

    def _semantic_chunk_text(self, cleaned: str) -> list[ChunkPayload]:
        """Chunk by adjacent sentence similarity using a lightweight TF-IDF signal."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                ChunkPayload(
                    level="local",
                    start_char=0,
                    end_char=len(sentences[0]),
                    approx_tokens=self._approx_tokens(sentences[0]),
                    content=sentences[0],
                )
            ]

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(sentences)
        adjacency_scores = []
        for idx in range(len(sentences) - 1):
            left = matrix[idx]
            right = matrix[idx + 1]
            numerator = left.multiply(right).sum()
            denom = np.linalg.norm(left.toarray()) * np.linalg.norm(right.toarray())
            adjacency_scores.append(float(numerator / denom) if denom else 0.0)

        threshold = (
            float(np.percentile(adjacency_scores, 25)) if adjacency_scores else 0.0
        )
        chunks: list[str] = []
        current = [sentences[0]]
        current_len = len(sentences[0])

        for idx, sentence in enumerate(sentences[1:], start=1):
            boundary_score = adjacency_scores[idx - 1]
            sentence_len = len(sentence) + 1
            if current and (
                boundary_score < threshold
                or current_len + sentence_len > self.config.chunk_max_chars
            ):
                chunks.append(" ".join(current).strip())
                if self.config.chunk_overlap > 0 and current:
                    overlap_chars = 0
                    overlap_sentences: list[str] = []
                    for prev in reversed(current):
                        overlap_chars += len(prev) + 1
                        overlap_sentences.insert(0, prev)
                        if overlap_chars >= self.config.chunk_overlap:
                            break
                    current = overlap_sentences
                    current_len = sum(len(s) + 1 for s in current)
                else:
                    current = []
                    current_len = 0
            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current).strip())

        payloads: list[ChunkPayload] = []
        cursor = 0
        for chunk in chunks:
            start_char = cleaned.find(chunk, cursor)
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(chunk)
            payloads.append(
                ChunkPayload(
                    level="local",
                    start_char=start_char,
                    end_char=end_char,
                    approx_tokens=self._approx_tokens(chunk),
                    content=chunk,
                )
            )
            cursor = end_char

        if len(cleaned) > self.config.chunk_max_chars or len(payloads) > 1:
            payloads.append(
                ChunkPayload(
                    level="document",
                    start_char=0,
                    end_char=len(cleaned),
                    approx_tokens=self._approx_tokens(cleaned),
                    content=cleaned,
                )
            )
        return payloads
