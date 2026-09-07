"""
Database engine and session management for structured memory (Phase 1B).

Uses SQLAlchemy against whatever settings.database_url points to -- SQLite
for now per the approved roadmap, swappable to Postgres later by changing
only the .env value, since no other code in this module is SQLite-specific.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.memory.models import Base

# check_same_thread=False is required for SQLite when used with FastAPI's
# threaded request handling. This flag is a no-op for other database
# backends, so it's safe to leave in place when you swap to Postgres later.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables that don't exist yet. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees it's
    closed afterward, even if the request raises.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
