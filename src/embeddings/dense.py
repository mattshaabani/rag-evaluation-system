"""
src/embeddings/dense.py

Dense embedding model wrapper using sentence-transformers.
Converts text → fixed-length vectors capturing semantic meaning.

The model (all-MiniLM-L6-v2) produces 384-dimensional vectors.
It runs locally, no API key needed.

Usage:
    from src.embeddings.dense import DenseEmbedder
    embedder = DenseEmbedder()
    vectors  = embedder.embed(["text one", "text two"])
    # vectors.shape → (2, 384)
"""

import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseEmbedder:
    """
    Wraps sentence-transformers for dense vector embeddings.

    Why wrap instead of using SentenceTransformer directly?
        - Adds lazy loading (model loads only when first used)
        - Adds batching logic and logging
        - Makes the rest of the codebase model-agnostic
          (swap models by changing config, not code)
    """

    def __init__(self, model_name: str = settings.embedding.dense_model):
        self.model_name = model_name
        self._model     = None   # lazy load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading dense embedding model",
                        extra={"model": self.model_name})
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Args:
            texts:         List of strings to embed.
            batch_size:    How many texts to process at once.
                           Larger = faster but more RAM.
            show_progress: Show tqdm progress bar for large batches.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
            Each row is the embedding vector for one text.

        Example:
            vectors = embedder.embed(["hello", "world"])
            # vectors.shape → (2, 384)
            # vectors[0]    → embedding for "hello"
            # vectors[1]    → embedding for "world"
        """
        if not texts:
            return np.array([])

        logger.debug(f"Embedding texts", extra={
            "count":      len(texts),
            "batch_size": batch_size,
        })

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalize → cosine sim = dot product
        )

        logger.debug(f"Embedding complete", extra={
            "shape": str(vectors.shape),
        })

        return vectors

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string. Convenience wrapper.
        Returns 1D array of shape (embedding_dim,).
        """
        return self.embed([text])[0]

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        Since we normalize embeddings at encode time,
        this is just the dot product.

        Returns value between -1 and 1.
        1.0  = identical meaning
        0.0  = unrelated
        -1.0 = opposite (rare in practice)
        """
        return float(np.dot(vec_a, vec_b))