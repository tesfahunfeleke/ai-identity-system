"""Tests for the chunker (chunk_text / chunk_raw_chunk)."""

from app.memory.ingestion.chunker import chunk_raw_chunk, chunk_text
from app.memory.ingestion.models import RawChunk, SourceType


def test_short_text_returns_single_chunk():
    text = "This is a short paragraph that should not be split."
    chunks = chunk_text(text, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_on_paragraph_boundaries():
    # Build paragraphs that individually fit but together exceed max_tokens.
    paragraph = " ".join(["word"] * 100)  # ~133 tokens at 0.75 words/token
    text = "\n\n".join([paragraph] * 5)  # ~665 tokens total

    chunks = chunk_text(text, max_tokens=200)
    assert len(chunks) > 1
    # No chunk should wildly exceed the limit.
    for chunk in chunks:
        assert len(chunk.split()) <= 200 / 0.75 + 5  # small tolerance


def test_oversized_single_paragraph_falls_back_to_sentences():
    sentence = "This is one sentence that repeats. "
    huge_paragraph = sentence * 80  # no blank lines, way over any token budget
    chunks = chunk_text(huge_paragraph, max_tokens=50)
    assert len(chunks) > 1


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_raw_chunk_preserves_metadata():
    long_text = "\n\n".join([" ".join(["word"] * 100)] * 5)
    raw = RawChunk(
        text=long_text,
        source_type=SourceType.JOURNAL_TEXT,
        source_file="some/file.md",
        extra={"title": "test"},
    )

    pieces = chunk_raw_chunk(raw, max_tokens=200)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.source_type == SourceType.JOURNAL_TEXT
        assert piece.source_file == "some/file.md"
        assert piece.extra == {"title": "test"}


def test_chunk_raw_chunk_short_text_returns_unchanged():
    raw = RawChunk(text="short", source_type=SourceType.JOURNAL_TEXT, source_file="f.md")
    pieces = chunk_raw_chunk(raw, max_tokens=500)
    assert pieces == [raw]
