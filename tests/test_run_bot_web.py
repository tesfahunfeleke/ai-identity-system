import logging

from fastapi.testclient import TestClient

import run_bot_web


def test_health_and_root_endpoints():
    client = TestClient(run_bot_web.app)

    health_response = client.get("/health")
    root_response = client.get("/")

    assert health_response.json() == {"status": "ok", "bot": "running"}
    assert root_response.json() == {
        "service": "AI Identity Telegram Bot",
        "status": "running",
    }


def test_bot_crash_is_logged_and_does_not_escape(monkeypatch, caplog):
    class CrashingBot:
        def run(self):
            raise RuntimeError("polling failed")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "configured")
    monkeypatch.setattr(run_bot_web, "TelegramBot", CrashingBot)

    with caplog.at_level(logging.ERROR, logger="run_bot_web"):
        run_bot_web._run_bot()

    assert "Telegram bot crashed; web service will continue running." in caplog.text


def test_main_starts_daemon_thread_and_uses_render_port(monkeypatch):
    thread_options = {}
    started = []
    server_options = {}
    startup_actions = []

    class FakeThread:
        def __init__(self, **kwargs):
            thread_options.update(kwargs)

        def start(self):
            startup_actions.append("thread")
            started.append(True)

    def fake_uvicorn_run(app, **kwargs):
        startup_actions.append("server")
        assert app is run_bot_web.app
        server_options.update(kwargs)

    monkeypatch.setattr(run_bot_web.threading, "Thread", FakeThread)
    monkeypatch.setattr(run_bot_web.uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(
        run_bot_web, "_ensure_memory_populated", lambda: startup_actions.append("ingestion")
    )
    monkeypatch.setenv("PORT", "8765")

    run_bot_web.main()

    assert startup_actions == ["ingestion", "thread", "server"]
    assert started
    assert thread_options["daemon"] is True
    assert server_options["port"] == 8765
    assert server_options["host"] == "0.0.0.0"

    monkeypatch.delenv("PORT")
    server_options.clear()
    run_bot_web.main()

    assert server_options["port"] == 8001


def test_startup_ingests_only_when_chromadb_is_empty(monkeypatch, caplog, tmp_path):
    class FakeVectorStore:
        def __init__(self, count):
            self.chunk_count = count

        def count(self):
            return self.chunk_count

    class FakeMemoryService:
        def __init__(self, count):
            self.vector_store = FakeVectorStore(count)
            self.indexed = []

        def index_chunks(self, chunks):
            self.indexed.extend(chunks)
            return len(chunks)

    chunks = [object(), object()]
    service = FakeMemoryService(count=0)
    monkeypatch.setattr(run_bot_web, "SemanticMemoryService", lambda: service)
    training_file = tmp_path / "about_me.txt"
    training_file.touch()
    monkeypatch.setattr(run_bot_web, "TRAINING_FILE", training_file)
    monkeypatch.setattr(run_bot_web, "run_ingestion", lambda source_dir: chunks)

    with caplog.at_level(logging.INFO, logger="run_bot_web"):
        run_bot_web._ensure_memory_populated()

    assert service.indexed == chunks
    assert "Running first-time ingestion..." in caplog.text


def test_startup_skips_ingestion_when_chromadb_is_populated(monkeypatch, caplog):
    class FakeMemoryService:
        vector_store = type("FakeVectorStore", (), {"count": lambda self: 7})()

    monkeypatch.setattr(run_bot_web, "SemanticMemoryService", FakeMemoryService)

    def fail_if_ingested(source_dir):
        raise AssertionError("unexpected ingestion")

    monkeypatch.setattr(run_bot_web, "run_ingestion", fail_if_ingested)

    with caplog.at_level(logging.INFO, logger="run_bot_web"):
        run_bot_web._ensure_memory_populated()

    assert "ChromaDB already populated with 7 chunks" in caplog.text


def test_admin_ingest_endpoint(monkeypatch):
    monkeypatch.setattr(run_bot_web, "_ingest_training_data", lambda: 3)

    response = TestClient(run_bot_web.app).post("/admin/ingest")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks_ingested": 3}
