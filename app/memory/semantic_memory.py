import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.memory.embeddings import EmbeddingService
from app.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SemanticMemoryService:
    """High-level service handling indexing, recency-weighted similarity search, and hybrid retrieval."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def index_chunks(self, chunks: List[Any]) -> int:
        """Embeds and persists a list of text chunks into vector memory."""
        if not chunks:
            return 0

        # Handle different chunk formats
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # Handle both dict and object formats
            if isinstance(chunk, dict):
                chunk_id = chunk.get('chunk_id', f"chunk_{len(ids)}")
                text = chunk.get('text', '')
                metadata = {
                    "source_type": chunk.get('source_type', 'unknown'),
                    "original_date": chunk.get('original_date', ''),
                    "source_file": chunk.get('source_file', ''),
                }
            else:
                # Assume it's an object with attributes
                chunk_id = getattr(chunk, 'chunk_id', f"chunk_{len(ids)}")
                text = getattr(chunk, 'text', '')
                metadata = {
                    "source_type": getattr(chunk, 'source_type', 'unknown'),
                    "original_date": getattr(chunk, 'original_date', ''),
                    "source_file": getattr(chunk, 'source_file', ''),
                }
            
            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)

        embeddings = self.embedding_service.embed_documents(texts)

        self.vector_store.add_chunks(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        return len(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source_type: Optional[str] = None,
        apply_recency_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieves top relevant memory chunks with optional recency decay weighting."""
        if not query or not query.strip():
            return []

        query_vector = self.embedding_service.embed_text(query)
        where_filter = {"source_type": source_type} if source_type else None

        # Fetch extra candidates to re-rank with recency scoring
        fetch_k = k * 3 if apply_recency_boost else k
        raw_results = self.vector_store.query(
            query_embedding=query_vector, n_results=fetch_k, where_filter=where_filter
        )

        if not raw_results:
            return []

        if not apply_recency_boost:
            return raw_results[:k]

        ranked_results = self._apply_recency_ranking(raw_results)
        return ranked_results[:k]

    def _apply_recency_ranking(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Applies time-decay weighting to vector similarity scores."""
        now = datetime.utcnow()
        scored_results = []

        for item in results:
            base_similarity = item["similarity"]
            date_str = item["metadata"].get("original_date")

            decay_factor = 1.0
            if date_str:
                try:
                    item_date = datetime.fromisoformat(date_str)
                    days_old = max(0, (now - item_date).days)
                    # Exponential decay factor: ~10% loss per 365 days
                    decay_factor = 1.0 / (1.0 + (days_old / 365.0) * 0.15)
                except ValueError:
                    decay_factor = 1.0

            final_score = base_similarity * decay_factor
            item_copy = dict(item)
            item_copy["combined_score"] = round(final_score, 4)
            scored_results.append(item_copy)

        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_results
