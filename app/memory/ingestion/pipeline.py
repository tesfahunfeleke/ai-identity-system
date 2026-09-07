"""
Ingestion pipeline orchestrator.

This is the only place in the codebase that knows the full list of
registered loaders. Everything else (the API endpoint, future CLI commands)
calls run_ingestion() and gets back chunked, normalized data -- it never
needs to know text files and chat exports are handled differently.

To add a new source format: write a loader implementing BaseLoader, add one
line to _LOADERS below. Nothing else in the pipeline changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.memory.ingestion.base import BaseLoader

if TYPE_CHECKING:
    from app.memory.semantic import VectorStore
from app.memory.ingestion.chat_loader import ChatExportLoader
from app.memory.ingestion.chunker import chunk_raw_chunk
from app.memory.ingestion.models import RawChunk
from app.memory.ingestion.text_loader import TextLoader

logger = logging.getLogger(__name__)

# Registered loaders, in priority order. The first loader whose can_handle()
# returns True for a given file is used.
_LOADERS: list[BaseLoader] = [
    TextLoader(),
    ChatExportLoader(),
]


def discover_files(source_dir: Path) -> list[Path]:
    """Find all files under source_dir that at least one loader can handle."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    files = [p for p in source_dir.rglob("*") if p.is_file()]
    handled = [f for f in files if any(loader.can_handle(f) for loader in _LOADERS)]

    skipped = len(files) - len(handled)
    if skipped:
        logger.info("Skipping %d file(s) with no matching loader", skipped)

    return handled


def load_file(file_path: Path) -> list[RawChunk]:
    """Load a single file using the first loader that can handle it."""
    for loader in _LOADERS:
        if loader.can_handle(file_path):
            try:
                return loader.load(file_path)
            except Exception:
                logger.exception("Failed to load %s, skipping", file_path)
                return []
    logger.warning("No loader registered for %s, skipping", file_path)
    return []


def run_ingestion(
    source_dir: Path,
    max_tokens_per_chunk: int = 500,
    vector_store: "VectorStore | None" = None,
) -> list[RawChunk]:
    """
    Run the full ingestion pipeline over every file in source_dir:
    discover -> load -> chunk -> (optionally) embed into semantic memory.

    Args:
        source_dir:           Directory to scan for supported files.
        max_tokens_per_chunk: Target chunk size ceiling (approximate tokens).
        vector_store:         If provided, chunks are upserted into the vector
                              store after chunking (Phase 2 behaviour). Pass
                              None to skip embedding (e.g. in unit tests that
                              only need to check loading/chunking).

    Returns:
        The final list of RawChunks produced, same regardless of whether
        embedding was requested.
    """
    files = discover_files(source_dir)
    logger.info("Discovered %d ingestible file(s) in %s", len(files), source_dir)

    all_chunks: list[RawChunk] = []
    for file_path in files:
        raw_chunks = load_file(file_path)
        for raw_chunk in raw_chunks:
            all_chunks.extend(chunk_raw_chunk(raw_chunk, max_tokens=max_tokens_per_chunk))

    logger.info("Ingestion produced %d chunk(s) from %d file(s)", len(all_chunks), len(files))

    if vector_store is not None and all_chunks:
        embedded = vector_store.add_chunks(all_chunks)
        logger.info("Embedded %d chunk(s) into semantic memory", embedded)

    return all_chunks
