"""
Tests for VectorStore (TF-IDF semantic memory).

No network, no model downloads, no DLL dependencies — pure Python.
Each test gets an isolated store via the `tmp_store` fixture.
"""

import pytest
from datetime import datetime, timezone

from app.memory.ingestion.models import RawChunk, SourceType
from app.memory.semantic import VectorStore, _tokenize, _cosine, _build_tfidf_index


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _chunk(text, source_type=SourceType.JOURNAL_TEXT,
           source_file="test.md", date=None):
    return RawChunk(
        text=text,
        source_type=source_type,
        source_file=source_file,
        original_date=date,
    )


# --------------------------------------------------------------------------
# Unit tests for internal helpers
# --------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    tokens = _tokenize("Hello, World! Testing 123.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "testing" in tokens


def test_tokenize_drops_single_chars():
    tokens = _tokenize("a b c dog")
    assert "a" not in tokens
    assert "dog" in tokens


def test_cosine_identical_vectors_is_one():
    vec = {"coffee": 0.6, "morning": 0.8}
    assert abs(_cosine(vec, vec) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors_is_zero():
    a = {"coffee": 1.0}
    b = {"woodworking": 1.0}
    assert _cosine(a, b) == 0.0


def test_cosine_empty_vector_is_zero():
    assert _cosine({}, {"coffee": 1.0}) == 0.0


def test_build_tfidf_index_returns_normalised_vectors():
    docs = ["I love coffee.", "My hobby is woodworking.", "I live in Austin."]
    idf, vectors = _build_tfidf_index(docs)
    for vec in vectors:
        if vec:
            norm = sum(v * v for v in vec.values()) ** 0.5
            assert abs(norm - 1.0) < 1e-6


# --------------------------------------------------------------------------
# VectorStore: add / count / persist
# --------------------------------------------------------------------------

def test_add_chunks_increases_count(tmp_store):
    tmp_store.add_chunks([
        _chunk("I love hiking on weekends."),
        _chunk("My favorite season is autumn."),
    ])
    assert tmp_store.count() == 2


def test_add_empty_list_does_nothing(tmp_store):
    assert tmp_store.add_chunks([]) == 0
    assert tmp_store.count() == 0


def test_upsert_same_chunk_does_not_duplicate(tmp_store):
    c = _chunk("I work at a design studio.")
    tmp_store.add_chunks([c])
    tmp_store.add_chunks([c])   # same text + file = same ID = upsert
    assert tmp_store.count() == 1


def test_store_persists_to_json(tmp_path, monkeypatch):
    """Data written by one VectorStore instance survives to a new instance."""
    monkeypatch.setattr("app.core.config.settings.chroma_persist_dir", str(tmp_path))
    import app.memory.semantic as sem_mod
    sem_mod._store = None

    store_a = VectorStore()
    store_a.add_chunks([_chunk("Persisted entry about coffee.")])

    sem_mod._store = None
    store_b = VectorStore()     # fresh instance, same path
    assert store_b.count() == 1
    sem_mod._store = None


# --------------------------------------------------------------------------
# VectorStore: retrieve
# --------------------------------------------------------------------------

def test_retrieve_empty_store_returns_empty_list(tmp_store):
    assert tmp_store.retrieve("coffee") == []


def test_retrieve_returns_results(tmp_store):
    tmp_store.add_chunks([
        _chunk("I love drinking coffee every morning."),
        _chunk("My hobby is woodworking and making furniture."),
        _chunk("I live in Austin, Texas."),
    ])
    results = tmp_store.retrieve("coffee morning", k=3)
    assert len(results) >= 1


def test_retrieve_top_result_contains_query_terms(tmp_store):
    tmp_store.add_chunks([
        _chunk("I enjoy coffee every morning without fail."),
        _chunk("My favorite hobby is woodworking in the garage."),
        _chunk("Austin has great weather in autumn."),
    ])
    results = tmp_store.retrieve("coffee morning", k=1)
    assert "coffee" in results[0].text.lower()


def test_retrieve_respects_k(tmp_store):
    for i in range(10):
        tmp_store.add_chunks([_chunk(f"Journal entry number {i} about my daily life.")])
    results = tmp_store.retrieve("journal entry", k=3)
    assert len(results) <= 3


def test_retrieve_filters_by_source_type(tmp_store):
    tmp_store.add_chunks([
        _chunk("I love coffee.", source_type=SourceType.JOURNAL_TEXT, source_file="j.md"),
        _chunk("USER: Coffee?\nASSISTANT: Yes!", source_type=SourceType.CHAT_EXPORT, source_file="c.json"),
    ])
    results = tmp_store.retrieve("coffee", k=5, source_type="journal_text")
    assert all(r.source_type == "journal_text" for r in results)


def test_retrieve_result_has_all_expected_fields(tmp_store):
    date = datetime(2026, 3, 14, tzinfo=timezone.utc)
    tmp_store.add_chunks([_chunk("I live in Austin.", date=date)])
    results = tmp_store.retrieve("Austin", k=1)
    r = results[0]
    assert r.text
    assert r.source_type
    assert r.source_file
    assert r.score > 0
    assert r.original_date is not None


def test_retrieve_scores_are_positive_for_matching_chunks(tmp_store):
    tmp_store.add_chunks([_chunk("I drink coffee every morning.")])
    results = tmp_store.retrieve("coffee", k=5)
    assert all(r.score > 0 for r in results)


# --------------------------------------------------------------------------
# Recency weighting
# --------------------------------------------------------------------------

def test_recency_boosts_score_of_newer_chunk(tmp_store):
    old = datetime(2018, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 3, 1, tzinfo=timezone.utc)
    # Identical text → same TF-IDF similarity; recency should break the tie.
    tmp_store.add_chunks([
        _chunk("I enjoy morning coffee.", date=old, source_file="old.md"),
        _chunk("I enjoy morning coffee.", date=new, source_file="new.md"),
    ])
    results = tmp_store.retrieve("morning coffee", k=2, apply_recency=True)
    assert results[0].source_file == "new.md"


def test_recency_off_still_returns_results(tmp_store):
    tmp_store.add_chunks([_chunk("Some text about coffee.")])
    results = tmp_store.retrieve("coffee", k=1, apply_recency=False)
    assert len(results) == 1


# --------------------------------------------------------------------------
# delete_source
# --------------------------------------------------------------------------

def test_delete_source_removes_correct_chunks(tmp_store):
    tmp_store.add_chunks([
        _chunk("Entry from file A.", source_file="a.md"),
        _chunk("Entry from file B.", source_file="b.md"),
    ])
    tmp_store.delete_source("a.md")
    assert tmp_store.count() == 1
    results = tmp_store.retrieve("file A", k=5)
    assert all("file A" not in r.text for r in results)


def test_delete_nonexistent_source_does_not_raise(tmp_store):
    tmp_store.add_chunks([_chunk("Some text.", source_file="real.md")])
    tmp_store.delete_source("ghost.md")     # should not raise
    assert tmp_store.count() == 1
