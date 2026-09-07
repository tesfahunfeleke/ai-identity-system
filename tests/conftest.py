"""
Shared pytest fixtures.

The TF-IDF store needs no model or network — tests run fully offline.
The only shared concern is pointing the store at a temp directory so
tests never touch real persisted data.
"""

import pytest


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """
    A fresh VectorStore backed by a temp directory, isolated per test.
    Monkeypatches settings so the store writes its JSON file to tmp_path
    instead of data/. Also resets the module-level singleton between tests.
    """
    monkeypatch.setattr("app.core.config.settings.chroma_persist_dir", str(tmp_path))

    # Reset the singleton so each test gets a clean instance.
    import app.memory.semantic as sem_mod
    sem_mod._store = None

    from app.memory.semantic import VectorStore
    store = VectorStore()
    yield store
    store.clear()
    sem_mod._store = None
