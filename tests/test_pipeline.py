"""Tests for the ingestion pipeline orchestrator (discover_files, run_ingestion)."""

from pathlib import Path

import pytest

from app.memory.ingestion.pipeline import discover_files, run_ingestion

FIXTURES = Path(__file__).parent / "fixtures" / "sample_journals"


def test_discover_files_finds_all_supported_formats():
    files = discover_files(FIXTURES)
    suffixes = {f.suffix for f in files}
    assert ".md" in suffixes
    assert ".txt" in suffixes
    assert ".json" in suffixes


def test_discover_files_raises_on_missing_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_files(missing)


def test_discover_files_skips_unsupported_formats(tmp_path):
    (tmp_path / "ignored.pdf").write_text("not handled")
    (tmp_path / "notes.txt").write_text("My favorite color is blue.")
    files = discover_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "notes.txt"


def test_run_ingestion_produces_chunks_from_all_fixture_files():
    chunks = run_ingestion(FIXTURES, max_tokens_per_chunk=500)
    assert len(chunks) > 0

    source_files = {Path(c.source_file).name for c in chunks}
    assert "journal_march.md" in source_files
    assert "random_notes.txt" in source_files
    assert "chatgpt_export.json" in source_files
    assert "claude_export.json" in source_files
    assert "generic_chat.json" in source_files


def test_run_ingestion_handles_malformed_file_gracefully(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json")
    (tmp_path / "fine.txt").write_text("My favorite drink is tea.")

    # Should not raise -- the broken file is logged and skipped.
    chunks = run_ingestion(tmp_path)
    source_files = {Path(c.source_file).name for c in chunks}
    assert "fine.txt" in source_files
    assert "broken.json" not in source_files
