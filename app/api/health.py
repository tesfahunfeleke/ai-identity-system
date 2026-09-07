"""
Health check endpoint.

Trivial on purpose for Phase 0 — its only job is to prove the server is up
and importable end-to-end. Later phases add real routers here
(memory.py, chat.py, etc.) and main.py includes them the same way.
"""

import logging

from fastapi import APIRouter

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Liveness check. Returns 200 with basic app info if the server is up."""
    logger.debug("Health check requested")
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }
