"""
Structured logging setup.

Call configure_logging() once at startup (done in app/main.py). Everywhere
else in the codebase, get a logger with `logging.getLogger(__name__)` as
normal — this module just configures the root handler/format once.
"""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """
    Configure root logging. Uses a plain readable format for local dev;
    swap the formatter for a JSON one later (Phase 8) if you want to ship
    logs to an aggregator.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicate handlers if configure_logging() is ever called twice
    # (e.g. under --reload in some setups).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers so your own logs aren't drowned out.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
