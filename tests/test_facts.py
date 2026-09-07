"""
Tests for facts.py (write_fact, get_facts, get_fact_history).

Uses an isolated in-memory SQLite database per test, completely separate
from the real app database, so these tests never touch real data and can
run in parallel without interference.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.memory.facts import get_fact_history, get_facts, write_fact
from app.memory.models import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_write_fact_creates_new_fact(db):
    fact = write_fact(db, category="preference", key="coffee_order", value="flat white")
    assert fact.id is not None
    assert fact.value == "flat white"
    assert fact.superseded_by is None


def test_get_facts_returns_written_fact(db):
    write_fact(db, category="preference", key="coffee_order", value="flat white")
    results = get_facts(db, category="preference")
    assert len(results) == 1
    assert results[0].value == "flat white"


def test_writing_same_value_does_not_duplicate(db):
    write_fact(db, category="preference", key="coffee_order", value="flat white")
    write_fact(db, category="preference", key="coffee_order", value="flat white")

    active_facts = get_facts(db, category="preference", key="coffee_order")
    assert len(active_facts) == 1


def test_writing_changed_value_supersedes_old_fact(db):
    old_fact = write_fact(db, category="preference", key="coffee_order", value="flat white")
    new_fact = write_fact(db, category="preference", key="coffee_order", value="cold brew")

    # Only the new fact should be "active".
    active_facts = get_facts(db, category="preference", key="coffee_order")
    assert len(active_facts) == 1
    assert active_facts[0].value == "cold brew"

    # The old fact should still exist in history, pointing to the new one.
    db.refresh(old_fact)
    assert old_fact.superseded_by == new_fact.id


def test_get_fact_history_returns_full_chain_oldest_first(db):
    write_fact(db, category="preference", key="coffee_order", value="drip coffee")
    write_fact(db, category="preference", key="coffee_order", value="flat white")
    write_fact(db, category="preference", key="coffee_order", value="cold brew")

    history = get_fact_history(db, category="preference", key="coffee_order")
    assert [f.value for f in history] == ["drip coffee", "flat white", "cold brew"]


def test_get_facts_filters_by_category_only(db):
    write_fact(db, category="preference", key="coffee_order", value="flat white")
    write_fact(db, category="trait", key="employer", value="design studio")

    preferences = get_facts(db, category="preference")
    assert len(preferences) == 1
    assert preferences[0].key == "coffee_order"
