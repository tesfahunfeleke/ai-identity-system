"""
Read/write operations for structured memory facts (Phase 1B).

Two operations matter most:
  - write_fact(): inserts a new fact, or supersedes an existing one with the
    same (category, key) if the value has actually changed -- this is what
    prevents "favorite coffee order" from accumulating duplicate, stale rows.
  - get_facts(): the query function the Phase 4 personality layer will call
    to pull relevant facts into a prompt.
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.models import Fact

logger = logging.getLogger(__name__)


def write_fact(
    db: Session,
    category: str,
    key: str,
    value: str,
    confidence: float = 0.5,
    source_excerpt: str | None = None,
    source_chunk_id: str | None = None,
) -> Fact:
    """
    Insert a new fact. If an active (non-superseded) fact already exists
    for the same (category, key), it is superseded by the new one --
    unless the value is identical, in which case the existing fact is
    returned unchanged (no point creating a duplicate for the same value).
    """
    existing = db.scalar(
        select(Fact).where(
            Fact.category == category,
            Fact.key == key,
            Fact.superseded_by.is_(None),
        )
    )

    if existing is not None and existing.value == value:
        logger.debug("Fact %s:%s unchanged, skipping write", category, key)
        return existing

    new_fact = Fact(
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        source_excerpt=source_excerpt,
        source_chunk_id=source_chunk_id,
    )
    db.add(new_fact)
    db.flush()  # assigns new_fact.id without committing yet

    if existing is not None:
        existing.superseded_by = new_fact.id
        logger.info("Fact %s:%s superseded (%r -> %r)", category, key, existing.value, value)
    else:
        logger.info("Fact %s:%s created (%r)", category, key, value)

    db.commit()
    return new_fact


def get_facts(
    db: Session,
    category: str | None = None,
    key: str | None = None,
    include_superseded: bool = False,
) -> list[Fact]:
    """
    Query facts, optionally filtered by category and/or key. By default only
    active (non-superseded) facts are returned -- the current source of
    truth the personality layer should read from.
    """
    stmt = select(Fact)

    if category is not None:
        stmt = stmt.where(Fact.category == category)
    if key is not None:
        stmt = stmt.where(Fact.key == key)
    if not include_superseded:
        stmt = stmt.where(Fact.superseded_by.is_(None))

    stmt = stmt.order_by(Fact.updated_at.desc())
    return list(db.scalars(stmt))


def get_fact_history(db: Session, category: str, key: str) -> list[Fact]:
    """
    Return the full history of a fact (all superseded versions plus the
    current one), oldest first -- useful for debugging or for a future
    "how has my answer to this changed over time" feature.
    """
    facts = get_facts(db, category=category, key=key, include_superseded=True)
    return sorted(facts, key=lambda f: f.created_at)
