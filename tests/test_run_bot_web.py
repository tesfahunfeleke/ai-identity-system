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

    class FakeThread:
        def __init__(self, **kwargs):
            thread_options.update(kwargs)

        def start(self):
            started.append(True)

    def fake_uvicorn_run(app, **kwargs):
        assert app is run_bot_web.app
        server_options.update(kwargs)

    monkeypatch.setattr(run_bot_web.threading, "Thread", FakeThread)
    monkeypatch.setattr(run_bot_web.uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setenv("PORT", "8765")

    run_bot_web.main()

    assert started
    assert thread_options["daemon"] is True
    assert server_options["port"] == 8765
    assert server_options["host"] == "0.0.0.0"

    monkeypatch.delenv("PORT")
    server_options.clear()
    run_bot_web.main()

    assert server_options["port"] == 8001
