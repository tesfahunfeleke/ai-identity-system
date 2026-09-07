"""
Ingestion endpoint: triggers the Phase 1 pipeline (load -> chunk -> extract
-> write facts) over a directory of source files.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.memory.db import get_db
from app.memory.extraction import extract_facts
from app.memory.facts import write_fact
from app.memory.ingestion import run_ingestion
from app.memory.semantic import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class IngestionRequest(BaseModel):
    source_dir: str


class IngestionResult(BaseModel):
    files_discovered: int
    chunks_produced: int
    chunks_embedded: int
    facts_written: int


@router.post("/run", response_model=IngestionResult)
def run_ingestion_endpoint(
    request: IngestionRequest, db: Session = Depends(get_db)
) -> IngestionResult:
    """
    Run the full Phase 1+2 pipeline over every supported file in source_dir:
    discover -> load -> chunk -> embed into semantic memory -> extract facts
    -> write to structured memory.
    """
    source_path = Path(request.source_dir)
    vector_store = get_vector_store()

    try:
        chunks = run_ingestion(source_path, vector_store=vector_store)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    files_discovered = len({chunk.source_file for chunk in chunks})
    chunks_embedded = vector_store.count()

    facts_written = 0
    for chunk in chunks:
        candidates = extract_facts(chunk.text)
        for candidate in candidates:
            write_fact(
                db,
                category=candidate.category,
                key=candidate.key,
                value=candidate.value,
                confidence=candidate.confidence,
                source_excerpt=candidate.source_excerpt,
            )
            facts_written += 1

    logger.info(
        "Ingestion complete: %d files, %d chunks, %d embedded, %d facts written",
        files_discovered,
        len(chunks),
        chunks_embedded,
        facts_written,
    )

    return IngestionResult(
        files_discovered=files_discovered,
        chunks_produced=len(chunks),
        chunks_embedded=chunks_embedded,
        facts_written=facts_written,
    )
