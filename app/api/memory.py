from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.memory.semantic_memory import SemanticMemoryService

router = APIRouter()
memory_service = SemanticMemoryService()


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    source_type: Optional[str] = None


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results_count: int
    results: List[SearchResult]


@router.get("/stats")
async def get_memory_stats():
    try:
        count = memory_service.vector_store.count()
        return {
            "status": "online",
            "total_vector_chunks": count,
            "collection_name": memory_service.vector_store.collection_name,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/search")
async def search_memory(request: SearchRequest):
    try:
        results = memory_service.retrieve(
            query=request.query,
            k=request.k,
            source_type=request.source_type
        )
        return SearchResponse(
            query=request.query,
            results_count=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
