"""
Ingestion pipeline: turns raw journal files and chat exports into chunked,
normalized text ready for fact extraction and embedding.

Public entry point: run_ingestion(source_dir) -> list[RawChunk]
"""

from app.memory.ingestion.models import RawChunk, SourceType
from app.memory.ingestion.pipeline import discover_files, load_file, run_ingestion

__all__ = ["run_ingestion", "discover_files", "load_file", "RawChunk", "SourceType"]
