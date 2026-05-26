"""
src/vectorstore/base.py

Abstract base class defining the vector store interface.
All vector store implementations must implement these methods.

This means the rest of the codebase never imports ChromaDB or
Qdrant directly — only this interface. Switching vector stores
requires changing one line in config, nothing else.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.data.chunker import Chunk


# ─────────────────────────────────────────────
# Search result container
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    A single result from a vector store search.

    Attributes:
        chunk:    The retrieved chunk with its content and metadata.
        score:    Similarity score. Higher = more similar.
                  Range depends on the store and distance metric.
        rank:     Position in the result list (1-indexed).
    """
    chunk: Chunk
    score: float
    rank:  int

    def __repr__(self) -> str:
        preview = self.chunk.content[:60].replace("\n", " ")
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, preview='{preview}...')"


# ─────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────

class BaseVectorStore(ABC):
    """
    Abstract interface for vector stores.

    Any class inheriting from this MUST implement all
    @abstractmethod methods or Python raises TypeError at
    instantiation time — catching missing implementations early.

    Usage pattern:
        store = ChromaVectorStore()   # or QdrantVectorStore()
        store.add_chunks(chunks, vectors)
        results = store.search(query_vector, top_k=5)
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks:  list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Add chunks and their embedding vectors to the store.

        Args:
            chunks:  List of Chunk objects (content + metadata).
            vectors: Corresponding embedding vectors.
                     len(chunks) must equal len(vectors).
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k:        int = 5,
        filters:      dict | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to a query vector.

        Args:
            query_vector: The embedded query as a list of floats.
            top_k:        Number of results to return.
            filters:      Optional metadata filters.
                          e.g. {"file_type": "pdf"}

        Returns:
            List of SearchResult sorted by similarity descending.
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete all vectors and chunks from the store."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return statistics about the current collection.
        Must include at least: {"count": int, "collection": str}
        """
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        """Return True if the collection exists and has data."""
        ...