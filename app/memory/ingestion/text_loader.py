"""
Loader for plain text and markdown journal files (.txt, .md).

Design choice: if the file contains date-like headers (e.g. "## 2026-03-14"
or "# March 14, 2026"), we split into one RawChunk per dated section, since
that's a natural semantic boundary and gives us a real original_date per
entry. If no such headers are found, the whole file becomes a single chunk
and original_date falls back to the file's last-modified time.
"""

import re
from datetime import datetime
from pathlib import Path

from app.memory.ingestion.base import BaseLoader
from app.memory.ingestion.models import RawChunk, SourceType

# Matches markdown headers that look like dates, e.g.:
#   ## 2026-03-14
#   # March 14, 2026
#   ### 03/14/2026
_DATE_HEADER_RE = re.compile(
    r"^#{1,6}\s*("
    r"\d{4}-\d{2}-\d{2}"                                  # 2026-03-14
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                            # 03/14/2026
    r"|[A-Za-z]+ \d{1,2},? \d{4}"                          # March 14, 2026
    r")\s*$",
    re.MULTILINE,
)

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%B %d %Y")


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip().rstrip(",")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class TextLoader(BaseLoader):
    """Loads .txt and .md journal files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: Path) -> list[RawChunk]:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        matches = list(_DATE_HEADER_RE.finditer(content))

        if not matches:
            # No dated sections found -- whole file is one chunk.
            fallback_date = datetime.fromtimestamp(file_path.stat().st_mtime)
            return [
                RawChunk(
                    text=content,
                    source_type=SourceType.JOURNAL_TEXT,
                    source_file=str(file_path),
                    original_date=fallback_date,
                )
            ]

        # Split into sections, one per date header.
        chunks: list[RawChunk] = []
        for i, match in enumerate(matches):
            section_start = match.end()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[section_start:section_end].strip()
            if not section_text:
                continue

            parsed_date = _parse_date(match.group(1))
            chunks.append(
                RawChunk(
                    text=section_text,
                    source_type=SourceType.JOURNAL_TEXT,
                    source_file=str(file_path),
                    original_date=parsed_date,
                )
            )
        return chunks
