"""Run the Telegram bot alongside a small FastAPI web service."""

import logging
import os
import threading
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from app.bot.telegram_bot import TelegramBot
from app.memory.ingestion import run_ingestion
from app.memory.semantic_memory import SemanticMemoryService

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Identity Telegram Bot")
TRAINING_DIR = Path(__file__).resolve().parent / "data" / "training"
TRAINING_FILE = TRAINING_DIR / "about_me.txt"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "bot": "running"}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "AI Identity Telegram Bot", "status": "running"}


def _ingest_training_data(memory_service: SemanticMemoryService | None = None) -> int:
    if not TRAINING_FILE.is_file():
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE}")

    service = memory_service or SemanticMemoryService()
    chunks = run_ingestion(TRAINING_DIR)
    return service.index_chunks(chunks)


def _ensure_memory_populated() -> None:
    memory_service = SemanticMemoryService()
    chunk_count = memory_service.vector_store.count()

    if chunk_count:
        logger.info("ChromaDB already populated with %d chunks", chunk_count)
        return

    if not TRAINING_FILE.is_file():
        logger.warning(
            "ChromaDB is empty and training file %s does not exist; skipping ingestion",
            TRAINING_FILE,
        )
        return

    logger.info("Running first-time ingestion...")
    ingested_count = _ingest_training_data(memory_service)
    logger.info("First-time ingestion complete: %d chunks", ingested_count)


@app.post("/admin/ingest")
def ingest_training_data() -> dict[str, int | str]:
    try:
        chunks_ingested = _ingest_training_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "ok", "chunks_ingested": chunks_ingested}


def _run_bot() -> None:
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.error("TELEGRAM_BOT_TOKEN is not configured; Telegram bot will not start.")
        return

    try:
        TelegramBot().run()
    except Exception:
        logger.exception("Telegram bot crashed; web service will continue running.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _ensure_memory_populated()

    bot_thread = threading.Thread(
        target=_run_bot,
        name="telegram-bot",
        daemon=True,
    )
    bot_thread.start()

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
