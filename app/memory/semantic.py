"""
Semantic memory: TF-IDF vector store backed by a plain JSON file.

Why TF-IDF instead of sentence embeddings:
  - Zero build dependencies beyond scikit-learn (pure Python wheel, installs
    cleanly on Windows Python 3.13 with no DLL issues)
  - Weights YOUR specific vocabulary heavily -- personal journals contain
    recurring names, places, and phrases that TF-IDF surfaces well
  - Fully local, zero network calls, zero privacy risk
  - The retrieve() interface is identical to what a future embedding-based
    store would expose, so swapping in sentence-transformers later (Phase 3+,
    once dependency issues are resolved) requires changing only this file

Architecture:
  - Chunks and their metadata are stored in a JSON file (data/semantic_store.json)
  - On first retrieve() after any add_chunks(), the TF-IDF matrix is rebuilt
    in memory from the persisted JSON -- fast at personal scale (hundreds of
    chunks, not millions)
  - Persistence is append-safe: add_chunks() loads existing data, merges, saves

Limitations vs. embedding-based retrieval:
  - Cannot do semantic leaps ("espresso" won't match "coffee" unless both
    appear in the corpus together)
  - For personal journals this is rarely a problem in practice
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.memory.ingestion.models import RawChunk

logger = logging.getLogger(__name__)

_STORE_FILENAME = "semantic_store.json"


# ---------------------------------------------------------------------------
# Public result type (identical interface to the embedding-based version)
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """A chunk returned by retrieve(), enriched with its relevance score."""

    __slots__ = ("text", "source_type", "source_file", "original_date", "extra", "score")

    def __init__(self, text, source_type, source_file, original_date, extra, score):
        self.text = text
        self.source_type = source_type
        self.source_file = source_file
        self.original_date = original_date
        self.extra = extra
        self.score = score

    def __repr__(self):
        date_str = self.original_date.date().isoformat() if self.original_date else "unknown"
        return f"<RetrievedChunk score={self.score:.3f} date={date_str} src={self.source_type}>"


# ---------------------------------------------------------------------------
# TF-IDF helpers  (no external deps -- pure Python)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop very short tokens."""
    import re
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1]


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _build_tfidf_index(documents: list[str]):
    """
    Build TF-IDF vectors for all documents.
    Returns (idf dict, list of tf-idf dicts).
    """
    N = len(documents)
    tokenized = [_tokenize(d) for d in documents]

    # IDF: log((N+1) / (df+1)) + 1  (scikit-learn smooth variant)
    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (count + 1)) + 1.0 for t, count in df.items()}

    # TF-IDF vectors (L2-normalised)
    vectors = []
    for tokens in tokenized:
        tf = _compute_tf(tokens)
        vec = {t: tf[t] * idf.get(t, 0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})

    return idf, vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    # Dot product over the smaller set for speed
    if len(a) > len(b):
        a, b = b, a
    return sum(a[t] * b[t] for t in a if t in b)


def _recency_bonus(original_date: datetime | None) -> float:
    """Small score bonus for more recent content, capped at 0.2."""
    if original_date is None:
        return 0.0
    if original_date.tzinfo is None:
        original_date = original_date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_years = max(0.0, (now - original_date).days / 365.25)
    bonus = 0.05 / (age_years + 1.0)
    return min(bonus, 0.2)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    TF-IDF semantic memory store with JSON persistence.

    All state lives in `data/semantic_store.json`. The TF-IDF matrix is
    rebuilt in memory on demand (lazy) after any mutation.
    """

    def __init__(self) -> None:
        self._store_path = Path(settings.chroma_persist_dir) / _STORE_FILENAME
        self._records: list[dict] = []         # persisted chunk records
        self._tfidf_vectors: list[dict] = []   # in-memory TF-IDF vectors
        self._dirty = True                     # True = need to rebuild index
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()

    def _load(self):
        if self._store_path.exists():
            try:
                self._records = json.loads(self._store_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d chunk(s) from semantic store", len(self._records))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load semantic store (%s), starting fresh", exc)
                self._records = []
        else:
            self._records = []
        self._dirty = True
        self._loaded = True

    def _save(self):
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_index(self):
        if not self._dirty:
            return
        if not self._records:
            self._tfidf_vectors = []
            self._dirty = False
            return
        texts = [r["text"] for r in self._records]
        _, self._tfidf_vectors = _build_tfidf_index(texts)
        self._dirty = False
        logger.debug("TF-IDF index rebuilt (%d chunks)", len(self._records))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[RawChunk]) -> int:
        """
        Add or update chunks. Re-ingesting the same source file replaces
        its existing vectors (upsert by chunk ID = SHA256 of file+text).
        """
        if not chunks:
            return 0
        self._ensure_loaded()

        import hashlib
        existing_ids = {r["id"] for r in self._records}
        added = 0

        for chunk in chunks:
            chunk_id = hashlib.sha256(
                f"{chunk.source_file}::{chunk.text}".encode()
            ).hexdigest()[:32]

            record = {
                "id": chunk_id,
                "text": chunk.text,
                "source_type": chunk.source_type.value,
                "source_file": chunk.source_file,
                "original_date": chunk.original_date.isoformat() if chunk.original_date else None,
                "extra": chunk.extra,
            }

            if chunk_id in existing_ids:
                # Replace in-place (upsert semantics)
                self._records = [r if r["id"] != chunk_id else record for r in self._records]
            else:
                self._records.append(record)
                existing_ids.add(chunk_id)
                added += 1

        self._dirty = True
        self._save()
        logger.info("Added %d new chunk(s) to semantic store (%d total)", added, len(self._records))
        return len(chunks)  # consistent with original interface

    def delete_source(self, source_file: str) -> None:
        """Remove all chunks from a given source file."""
        self._ensure_loaded()
        before = len(self._records)
        self._records = [r for r in self._records if r["source_file"] != source_file]
        removed = before - len(self._records)
        if removed:
            self._dirty = True
            self._save()
            logger.info("Deleted %d chunk(s) for source: %s", removed, source_file)

    def clear(self) -> None:
        """Wipe the entire store. Used in tests."""
        self._records = []
        self._tfidf_vectors = []
        self._dirty = False
        if self._store_path.exists():
            self._store_path.unlink()
        logger.warning("Semantic memory cleared")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source_type: str | None = None,
        apply_recency: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Return the top-k most relevant chunks for query using TF-IDF cosine
        similarity, with an optional small recency bonus.
        """
        self._ensure_loaded()
        if not self._records:
            return []

        self._rebuild_index()

        # Build query vector against the corpus IDF
        texts = [r["text"] for r in self._records]
        idf, _ = _build_tfidf_index(texts)   # fast at personal scale
        q_tokens = _tokenize(query)
        q_tf = _compute_tf(q_tokens)
        q_vec = {t: q_tf[t] * idf.get(t, 0) for t in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        results = []
        for record, doc_vec in zip(self._records, self._tfidf_vectors):
            if source_type and record["source_type"] != source_type:
                continue

            sim = _cosine(q_vec, doc_vec)

            original_date = None
            if record.get("original_date"):
                try:
                    original_date = datetime.fromisoformat(record["original_date"])
                except ValueError:
                    pass

            bonus = _recency_bonus(original_date) if apply_recency else 0.0

            results.append(RetrievedChunk(
                text=record["text"],
                source_type=record["source_type"],
                source_file=record["source_file"],
                original_date=original_date,
                extra=record.get("extra", {}),
                score=sim + bonus,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._records)


# Process-wide singleton
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
