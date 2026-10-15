"""Run the Telegram bot alongside a small FastAPI web service."""

import logging
import os
import threading

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.bot.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Identity Telegram Bot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "bot": "running"}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "AI Identity Telegram Bot", "status": "running"}


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
