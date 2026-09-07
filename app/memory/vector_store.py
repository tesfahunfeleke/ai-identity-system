import os
import logging
from typing import List, Dict, Any, Optional

# Try to import chromadb, but don't fail if not available
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store client for managing persistent semantic collections."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "semantic_memory",
    ):
        self.persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = collection_name

        if chromadb:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(allow_reset=True, anonymized_telemetry=False) if Settings else None,
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            self.available = True
            logger.info(f"Initialized ChromaDB collection '{self.collection_name}' at {self.persist_dir}")
        else:
            self.client = None
            self.collection = None
            self.available = False
            logger.warning("ChromaDB not installed. VectorStore running in mock mode.")

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert embedded text chunks into ChromaDB."""
        if not self.available:
            logger.info(f"Mock mode: Would upsert {len(ids)} chunks")
            return

        if not ids or not embeddings or not documents:
            logger.warning("Empty payload provided to add_chunks. Skipping.")
            return

        formatted_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            formatted_metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=formatted_metadatas,
        )
        logger.info(f"Successfully upserted {len(ids)} chunks to vector store.")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query nearest vector neighbors with optional metadata filtering."""
        if not self.available:
            return []

        if self.collection.count() == 0:
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self.collection.count())),
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        items = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for idx in range(len(ids)):
                dist = distances[idx] if idx < len(distances) else 1.0
                similarity = max(0.0, 1.0 - dist)
                items.append(
                    {
                        "chunk_id": ids[idx],
                        "text": docs[idx] if idx < len(docs) else "",
                        "metadata": metas[idx] if idx < len(metas) else {},
                        "distance": dist,
                        "similarity": round(similarity, 4),
                    }
                )

        return items

    def count(self) -> int:
        """Returns total records in the collection."""
        if not self.available:
            return 0
        return self.collection.count()

    def reset(self) -> None:
        """Clears the collection."""
        if not self.available:
            return
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
