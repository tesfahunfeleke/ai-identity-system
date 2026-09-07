import os
import logging
import hashlib
from typing import List, Optional

# Try to import OpenAI, but don't fail if not available
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or local fallback."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if api_key and OpenAI:
            self.client = OpenAI(api_key=api_key)
            self.available = True
        else:
            self.client = None
            self.available = False
            logger.warning(
                "OPENAI_API_KEY not found or OpenAI not installed. "
                "EmbeddingService running in mock/fallback mode."
            )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not text or not text.strip():
            raise ValueError("Text for embedding cannot be empty.")

        if self.available and self.client:
            try:
                response = self.client.embeddings.create(
                    input=text, model=self.model_name
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating embedding from OpenAI: {e}")
                # Fall back to mock
                return self._mock_embedding(text)
        else:
            # Fallback deterministic pseudo-embedding for testing/local offline dev
            return self._mock_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of strings."""
        if not texts:
            return []

        cleaned_texts = [t if t.strip() else " " for t in texts]

        if self.available and self.client:
            try:
                response = self.client.embeddings.create(
                    input=cleaned_texts, model=self.model_name
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"Error generating batch embeddings from OpenAI: {e}")
                # Fall back to mock
                return [self._mock_embedding(t) for t in cleaned_texts]
        else:
            return [self._mock_embedding(t) for t in cleaned_texts]

    def _mock_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generates a deterministic vector of specified dimension for offline tests."""
        hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(dim):
            byte_val = hash_digest[i % len(hash_digest)]
            norm_val = (byte_val / 255.0) * 2.0 - 1.0
            vec.append(round(norm_val, 6))
        return vec
