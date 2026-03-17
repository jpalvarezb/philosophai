"""Load source corpus files into DuckDB."""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import IngestConfig

if TYPE_CHECKING:
    from ..storage import DuckDBStorage


logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class ResourceFile:
    """Metadata parsed from a resource file path."""

    text_id: int
    file_path: str
    title: str
    author_source: str | None
    tradition: str | None = None
    domains: str | None = None
    time_period: str | None = None
    file_ext: str | None = None
    file_size_bytes: int | None = None


def _collapse_whitespace(text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _strip_html(markup: str) -> str:
    cleaned = TAG_RE.sub(" ", markup)
    return _collapse_whitespace(html.unescape(cleaned))


def parse_resource_metadata(path: Path, fallback_id: int) -> ResourceFile:
    """Infer title/author metadata from the resource filename."""
    stem_parts = path.stem.split("_")
    text_id = fallback_id
    title_parts = stem_parts
    author_source: str | None = None

    if stem_parts and stem_parts[0].isdigit():
        text_id = int(stem_parts[0]) + 1
        title_parts = stem_parts[1:]

    if len(title_parts) >= 2:
        author_source = title_parts[-1].replace("-", " ").strip() or None
        title = " ".join(part.replace("-", " ").strip() for part in title_parts[:-1]).strip()
    else:
        title = path.stem.replace("_", " ").replace("-", " ").strip()

    return ResourceFile(
        text_id=text_id,
        file_path=str(path),
        title=title or path.stem,
        author_source=author_source,
        file_ext=path.suffix.lower(),
        file_size_bytes=path.stat().st_size if path.exists() else None,
    )


class CorpusLoader:
    """Load the hard corpus resources into the DB."""

    def __init__(self, storage: "DuckDBStorage", config: IngestConfig | None = None):
        self.storage = storage
        self.config = config or IngestConfig()

    def ensure_schema(self) -> None:
        """Create the raw corpus tables if they do not exist."""
        con = self.storage.con
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                text_id INTEGER PRIMARY KEY,
                file_path VARCHAR UNIQUE,
                title VARCHAR,
                author_source VARCHAR,
                tradition VARCHAR,
                domains VARCHAR,
                time_period VARCHAR,
                file_ext VARCHAR,
                file_size_bytes BIGINT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_texts (
                text_id INTEGER PRIMARY KEY,
                content VARCHAR,
                char_count INTEGER,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def list_resource_files(self) -> list[Path]:
        """Discover files under the configured resources directory."""
        resources_dir = Path(self.config.resources_dir)
        if not resources_dir.exists():
            raise FileNotFoundError(f"Resources directory not found: {resources_dir}")

        allowed = {ext.lower() for ext in self.config.supported_formats}
        files = [
            path
            for path in sorted(resources_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in allowed
        ]
        logger.info("Discovered %s resource files in %s", len(files), resources_dir)
        return files

    def load_resources(self, skip_existing: bool = True) -> dict[str, int]:
        """Load source files and raw text into DuckDB."""
        self.ensure_schema()
        con = self.storage.con
        loaded = 0
        skipped = 0

        files = self.list_resource_files()
        total_files = len(files)
        for idx, path in enumerate(files, start=1):
            meta = parse_resource_metadata(path, fallback_id=idx)

            existing = con.execute(
                "SELECT text_id FROM files WHERE file_path = ?",
                [meta.file_path],
            ).fetchone()
            if existing and skip_existing:
                skipped += 1
                continue
            if existing:
                meta.text_id = int(existing[0])

            logger.info("[%d/%d] Loading %s …", idx, total_files, path.name)
            raw_text = self.load_text(path)
            if not raw_text.strip():
                skipped += 1
                continue

            con.execute(
                """
                INSERT INTO files (
                    text_id, file_path, title, author_source, tradition, domains,
                    time_period, file_ext, file_size_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (text_id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    title = EXCLUDED.title,
                    author_source = EXCLUDED.author_source,
                    tradition = EXCLUDED.tradition,
                    domains = EXCLUDED.domains,
                    time_period = EXCLUDED.time_period,
                    file_ext = EXCLUDED.file_ext,
                    file_size_bytes = EXCLUDED.file_size_bytes
                """,
                [
                    meta.text_id,
                    meta.file_path,
                    meta.title,
                    meta.author_source,
                    meta.tradition,
                    meta.domains,
                    meta.time_period,
                    meta.file_ext,
                    meta.file_size_bytes,
                ],
            )
            con.execute(
                """
                INSERT INTO raw_texts (text_id, content, char_count)
                VALUES (?, ?, ?)
                ON CONFLICT (text_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    char_count = EXCLUDED.char_count
                """,
                [meta.text_id, raw_text, len(raw_text)],
            )
            loaded += 1

        return {
            "discovered": len(files),
            "loaded": loaded,
            "skipped": skipped,
            "files_in_table": con.execute("SELECT COUNT(*) FROM files").fetchone()[0],
        }

    def load_text(self, path: Path) -> str:
        """Dispatch to the right loader based on suffix."""
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self.load_txt(path)
        if suffix == ".pdf":
            return self.load_pdf(path, config=self.config)
        if suffix == ".epub":
            return self.load_epub(path)
        raise ValueError(f"Unsupported resource format: {suffix}")

    @staticmethod
    def load_txt(path: Path) -> str:
        """Read plain text resources with fallback encodings."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    @staticmethod
    def load_pdf(path: Path, config: IngestConfig | None = None) -> str:
        """Extract text from PDF resources, with optional OCR fallback."""
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - dependency-driven
            raise ImportError("pymupdf is required to load PDF resources") from exc

        config = config or IngestConfig()
        text_parts: list[str] = []
        ocr_pages = 0

        with fitz.open(path) as doc:
            total_pages = len(doc)
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(page_text)
                    continue

                if not config.ocr_enabled:
                    continue

                try:
                    tp = page.get_textpage_ocr(
                        language=config.ocr_language,
                        dpi=config.ocr_dpi,
                    )
                    ocr_text = page.get_text("text", textpage=tp)
                    if ocr_text.strip():
                        text_parts.append(ocr_text)
                        ocr_pages += 1
                        if ocr_pages % 50 == 0:
                            logger.info(
                                "  OCR progress %s: %d/%d pages",
                                path.name,
                                page_num + 1,
                                total_pages,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("OCR failed on page %d of %s: %s", page_num, path.name, exc)

        result = "\n".join(part for part in text_parts if part).strip()

        if not result and not config.ocr_enabled:
            has_images = False
            with fitz.open(path) as doc:
                for page in doc:
                    if page.get_images():
                        has_images = True
                        break
            if has_images:
                logger.warning(
                    "Scanned/image PDF with no text layer: %s (%d pages). "
                    "Re-run with --ocr to extract via Tesseract.",
                    path.name,
                    total_pages,
                )
            else:
                logger.warning("PDF yielded no text: %s", path.name)
        elif ocr_pages:
            logger.info("OCR extracted text from %d/%d pages of %s", ocr_pages, total_pages, path.name)

        return result

    @staticmethod
    def load_epub(path: Path) -> str:
        """Extract text from EPUB resources."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError as exc:  # pragma: no cover - dependency-driven
            raise ImportError("ebooklib is required to load EPUB resources") from exc

        book = epub.read_epub(str(path))
        sections: list[str] = []
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            body = item.get_body_content()
            if not body:
                continue
            sections.append(_strip_html(body.decode("utf-8", errors="ignore")))
        return "\n\n".join(section for section in sections if section).strip()
