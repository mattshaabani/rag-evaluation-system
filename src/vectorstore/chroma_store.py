"""
src/vectorstore/chroma_store.py

ChromaDB vector store implementation.
ChromaDB is an embedded vector database — it runs inside your
Python process, no separate server needed in development.

In production mode (docker-compose) it runs as a separate service.
We support both modes via the host/port config.

Usage:
    from src.vectorstore.chroma_store import ChromaVectorStore
    store = ChromaVectorStore()
    store.add_chunks(chunks, vectors)
    results = store.search(query_vector, top_k=5)
"""

import uuid
import numpy as np

from src.vectorstore.base import BaseVectorStore, SearchResult
from src.data.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of BaseVectorStore.

    ChromaDB stores:
        - The raw text (documents)
        - The embedding vectors (embeddings)
        - Arbitrary metadata (metadatas)
        - A unique ID per entry (ids)

    All four must be provided together and stay in sync.
    """

    def __init__(
        self,
        collection_name: str  = settings.vectorstore.collection_name,
        host:            str  = settings.env.chroma_host,
        port:            int  = settings.env.chroma_port,
        in_memory:       bool = True,
    ):
        """
        Args:
            collection_name: Name of the ChromaDB collection.
            host/port:       For connecting to a remote ChromaDB server.
            in_memory:       If True, use local in-memory client (dev mode).
                             If False, connect to a running ChromaDB server.
        """
        self.collection_name = collection_name
        self.in_memory       = in_memory
        self._client         = None
        self._collection     = None

        logger.info(f"Initializing ChromaVectorStore", extra={
            "collection": collection_name,
            "mode":       "in_memory" if in_memory else f"{host}:{port}",
        })

    @property
    def client(self):
        """Lazy-load the ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("Run: pip install chromadb")

            if self.in_memory:
                self._client = chromadb.Client()
            else:
                self._client = chromadb.HttpClient(
                    host=settings.env.chroma_host,
                    port=settings.env.chroma_port,
                )
        return self._client

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # cosine distance for similarity search
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(
        self,
        chunks:  list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) "
                f"must have the same length"
            )

        if not chunks:
            logger.warning("add_chunks called with empty list")
            return

        # ChromaDB requires string IDs
        ids        = [str(uuid.uuid4()) for _ in chunks]
        documents  = [chunk.content for chunk in chunks]

        # ChromaDB metadata values must be str, int, float, or bool
        # Convert any other types (like Path) to string
        metadatas = []
        for chunk in chunks:
            clean_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        # Convert numpy arrays to lists (ChromaDB needs plain Python lists)
        vectors_list = [
            v.tolist() if isinstance(v, np.ndarray) else v
            for v in vectors
        ]

        # ChromaDB add — all four lists must be same length
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors_list,
            metadatas=metadatas,
        )

        logger.info(f"Added chunks to ChromaDB", extra={
            "count":      len(chunks),
            "collection": self.collection_name,
        })

    def search(
        self,
        query_vector: list[float],
        top_k:        int = settings.retrieval.top_k,
        filters:      dict | None = None,
    ) -> list[SearchResult]:
        """
        Search using ChromaDB's built-in HNSW index.

        ChromaDB returns results with 'distances' which are
        cosine distances (0=identical, 2=opposite).
        We convert to similarity scores (1=identical, -1=opposite).
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results":        min(top_k, self.collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }

        # Add metadata filter if provided
        if filters:
            query_kwargs["where"] = filters

        raw = self.collection.query(**query_kwargs)

        results = []
        for rank, (doc, meta, dist) in enumerate(zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ), start=1):
            # Convert cosine distance → similarity score
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # We want: 1=identical, -1=opposite
            similarity = 1.0 - dist

            chunk = Chunk(content=doc, metadata=meta)
            results.append(SearchResult(chunk=chunk, score=similarity, rank=rank))

        logger.debug(f"ChromaDB search complete", extra={
            "top_k":   top_k,
            "results": len(results),
        })

        return results

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Deleted collection", extra={"collection": self.collection_name})

    def get_stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "count":      self.collection.count(),
            "backend":    "chromadb",
        }

    def collection_exists(self) -> bool:
        try:
            return self.collection.count() > 0
        except Exception:
            return False