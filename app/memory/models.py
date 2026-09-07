"""
SQLAlchemy models for structured memory (Phase 1B).

This is the "facts" table from the roadmap: queryable, updatable knowledge
about the user, distinct from semantic memory (Phase 2's vector store of
raw journal/chat excerpts).
"""

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FactCategory(str, Enum):
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    BELIEF = "belief"
    TRAIT = "trait"


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Fact(Base):
    """A single piece of structured knowledge about the user."""

    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)

    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    source_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Self-referential: when a fact changes (e.g. favorite coffee order
    # updates), the old row is kept for history and points to its replacement
    # rather than being deleted or duplicated.
    superseded_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("facts.id"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Fact {self.category}:{self.key}={self.value!r}>"
