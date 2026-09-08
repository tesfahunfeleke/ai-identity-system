"""
FastAPI application entrypoint.
Run with: uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import health, ingestion, search, chat, memory, proactive
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


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return RedirectResponse(url="/static/index.html")


# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }


# Health endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Identity System is running"}


@app.get("/health/health")
async def health_check_alt():
    return {"status": "healthy", "message": "AI Identity System is running"}


# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(chat.router)
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(proactive.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
