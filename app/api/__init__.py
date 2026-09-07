from fastapi import APIRouter

from . import health, ingestion, search, chat, memory, proactive

# Main API router
router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
router.include_router(search.router, prefix="/search", tags=["search"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(memory.router, prefix="/memory", tags=["memory"])
router.include_router(proactive.router)
