"""
Semantic search endpoint: retrieve relevant personal memory chunks for a query.

This is the Phase 2 public API. The Phase 4 personality layer will call the
same VectorStore.retrieve() function internally when building prompts -- this
endpoint exists so you can explore/debug the retrieval quality interactively
from /docs before Phase 4 exists.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.memory.semantic import RetrievedChunk, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or message to search for")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    source_type: str | None = Field(
        default=None,
        description="Filter by source type: 'journal_text' or 'chat_export'",
    )
    apply_recency: bool = Field(
        default=True,
        description="Whether to give a small score boost to more recent content",
    )


class SearchResultItem(BaseModel):
    text: str
    source_type: str
    source_file: str
    original_date: str | None
    score: float
    extra: dict


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResultItem]


def _format_result(chunk: RetrievedChunk) -> SearchResultItem:
    return SearchResultItem(
        text=chunk.text,
        source_type=chunk.source_type,
        source_file=chunk.source_file,
        original_date=chunk.original_date.isoformat() if chunk.original_date else None,
        score=round(chunk.score, 4),
        extra=chunk.extra,
    )


@router.post("/", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    """
    Retrieve the most semantically relevant chunks from personal memory
    for the given query. Results are ranked by cosine similarity to the
    query embedding, with an optional small recency bonus.
    """
    vector_store = get_vector_store()
    results = vector_store.retrieve(
        query=request.query,
        k=request.k,
        source_type=request.source_type,
        apply_recency=request.apply_recency,
    )

    logger.info(
        "Search '%s' -> %d result(s) from %d total vectors",
        request.query[:60],
        len(results),
        vector_store.count(),
    )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=[_format_result(r) for r in results],
    )
