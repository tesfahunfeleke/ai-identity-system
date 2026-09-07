"""
Common data shapes for the ingestion pipeline.

Every loader (text, chat export, future sources) must produce a list of
RawChunk objects. This is the contract that lets the rest of the pipeline
(chunking, embedding in Phase 2, fact extraction in 1B) stay completely
ignorant of where the data originally came from.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Where a piece of raw text originally came from."""

    JOURNAL_TEXT = "journal_text"   # .txt / .md journal entries
    CHAT_EXPORT = "chat_export"     # ChatGPT/Claude/etc. JSON exports


@dataclass
class RawChunk:
    """
    A single unit of source text after loading, before chunking.

    One loaded file may produce multiple RawChunks (e.g. a chat export with
    many conversations, or a long journal file split by date headers).
    This is intentionally "raw" — chunking to embedding-sized pieces happens
    later in chunker.py, kept separate so loaders stay simple.
    """

    text: str
    source_type: SourceType
    source_file: str                       # path the data came from, for traceability
    original_date: datetime | None = None  # when the content was created/written
    extra: dict = field(default_factory=dict)  # loader-specific metadata (e.g. role, title)
