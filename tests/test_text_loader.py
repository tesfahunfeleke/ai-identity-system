"""Tests for TextLoader (.txt / .md files)."""

from pathlib import Path

from app.memory.ingestion.models import SourceType
from app.memory.ingestion.text_loader import TextLoader

FIXTURES = Path(__file__).parent / "fixtures" / "sample_journals"


def test_can_handle_txt_and_md():
    loader = TextLoader()
    assert loader.can_handle(Path("notes.txt"))
    assert loader.can_handle(Path("notes.md"))
    assert not loader.can_handle(Path("notes.json"))


def test_splits_dated_markdown_into_multiple_chunks():
    loader = TextLoader()
    chunks = loader.load(FIXTURES / "journal_march.md")

    assert len(chunks) == 3
    assert all(c.source_type == SourceType.JOURNAL_TEXT for c in chunks)
    assert all(c.original_date is not None for c in chunks)
    # Chunks should be in chronological order matching the file's headers.
    dates = [c.original_date.date().isoformat() for c in chunks]
    assert dates == ["2026-03-10", "2026-03-14", "2026-03-20"]


def test_dated_chunk_contains_expected_text():
    loader = TextLoader()
    chunks = loader.load(FIXTURES / "journal_march.md")
    middle_chunk = chunks[1]
    assert "oat milk flat white" in middle_chunk.text


def test_undated_text_file_becomes_single_chunk():
    loader = TextLoader()
    chunks = loader.load(FIXTURES / "random_notes.txt")

    assert len(chunks) == 1
    assert chunks[0].source_type == SourceType.JOURNAL_TEXT
    assert "woodworking" in chunks[0].text
    # No date header present -- should fall back to file mtime, not be None.
    assert chunks[0].original_date is not None


def test_empty_file_produces_no_chunks(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n  ")
    loader = TextLoader()
    assert loader.load(empty_file) == []
