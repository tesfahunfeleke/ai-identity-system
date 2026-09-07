"""
FastAPI application entrypoint.
Run with: uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api import health, ingestion, search, chat, proactive
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.memory.db import init_db

configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s (env=%s, debug=%s)",
        settings.app_name,
        settings.app_env,
        settings.debug,
    )
    init_db()
    logger.info("Structured memory database ready at %s", settings.database_url)
    yield
    logger.info("Shutting down %s", settings.app_name)

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])   # ← New
app.include_router(proactive.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)