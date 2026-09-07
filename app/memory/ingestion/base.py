"""
Abstract loader interface.

Every source format (text/markdown, chat exports, and anything added later —
Slack, Notion, email, whatever) implements this interface. The pipeline
(pipeline.py) only ever talks to loaders through this contract, so it never
needs to know which concrete loader it's running.

To add a new source: subclass BaseLoader, implement can_handle() and load(),
register it in pipeline.py's loader list. Nothing else changes.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.memory.ingestion.models import RawChunk


class BaseLoader(ABC):
    """Common interface for all source-format loaders."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Return True if this loader knows how to read the given file."""
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path: Path) -> list[RawChunk]:
        """
        Read the file and return one or more RawChunk objects.

        Implementations should be defensive: a malformed file should raise
        a clear exception rather than silently returning partial/garbage data,
        since fact extraction downstream trusts this output.
        """
        raise NotImplementedError
