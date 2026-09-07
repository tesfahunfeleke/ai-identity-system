"""
Splits RawChunks into smaller, semantically coherent pieces sized for later
embedding (Phase 2) and fact extraction (Phase 1B).

Target: ~200-500 tokens per chunk, per the roadmap. We approximate token
count with a simple word-count heuristic (~0.75 tokens/word for English) to
avoid pulling in a tokenizer dependency this early -- swap for a real
tokenizer (e.g. tiktoken) in Phase 2 if precision matters more there.

Splitting strategy: prefer paragraph boundaries (blank lines) so we don't
cut sentences in half. If a single paragraph alone exceeds the max size
(rare, but happens with chat transcripts), fall back to sentence-level
splitting within that paragraph.
"""

import re

from app.memory.ingestion.models import RawChunk

_WORDS_PER_TOKEN = 0.75  # rough heuristic: 1 token ~= 0.75 words for English
_TARGET_MIN_TOKENS = 200
_TARGET_MAX_TOKENS = 500

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    return int(word_count / _WORDS_PER_TOKEN) if word_count else 0


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Fall back to sentence-level splitting for an oversized paragraph."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, max_tokens: int = _TARGET_MAX_TOKENS) -> list[str]:
    """
    Split a block of text into pieces of roughly max_tokens or fewer,
    preferring to break on paragraph boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)

        # A single paragraph too big on its own: split it directly.
        if paragraph_tokens > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(_split_long_paragraph(paragraph, max_tokens))
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_raw_chunk(raw_chunk: RawChunk, max_tokens: int = _TARGET_MAX_TOKENS) -> list[RawChunk]:
    """
    Apply chunk_text() to a RawChunk's text, preserving all of its metadata
    across the resulting pieces. If the chunk is already small enough, it's
    returned unchanged (still as a one-item list, for a consistent call site).
    """
    if _estimate_tokens(raw_chunk.text) <= max_tokens:
        return [raw_chunk]

    pieces = chunk_text(raw_chunk.text, max_tokens=max_tokens)
    return [
        RawChunk(
            text=piece,
            source_type=raw_chunk.source_type,
            source_file=raw_chunk.source_file,
            original_date=raw_chunk.original_date,
            extra=dict(raw_chunk.extra),
        )
        for piece in pieces
    ]
