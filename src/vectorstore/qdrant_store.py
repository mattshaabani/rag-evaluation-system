"""
src/vectorstore/qdrant_store.py

Qdrant vector store implementation.
Qdrant is a dedicated high-performance vector database.

Key difference from ChromaDB:
    ChromaDB: simple, embedded, great for development
    Qdrant:   production-grade, supports filtering, payloads,
              distributed deployment, higher throughput

Both implement the same BaseVectorStore interface so the
rest of the code doesn't care which one is used.

Usage:
    from src.vectorstore.qdrant_store import QdrantVectorStore
    store = QdrantVectorStore(vector_size=384)
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


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation of BaseVectorStore.

    Qdrant concepts:
        Collection: equivalent to a table — stores vectors + payloads
        Point:      one entry = vector + payload (metadata) + id
        Payload:    arbitrary JSON metadata attached to each point
        Distance:   how similarity is measured (Cosine, Dot, Euclid)

    Why Cosine distance?
        Our embeddings are L2-normalized (unit vectors).
        For unit vectors: cosine_similarity = dot_product
        Cosine distance is rotation-invariant — scale doesn't matter,
        only direction (meaning) matters.
    """

    def __init__(
        self,
        collection_name: str = settings.vectorstore.collection_name,
        vector_size:     int = 384,
        host:            str = settings.env.qdrant_host,
        port:            int = settings.env.qdrant_port,
        in_memory:       bool = True,
    ):
        """
        Args:
            vector_size: Dimensionality of embedding vectors.
                         Must match your embedding model output.
                         all-MiniLM-L6-v2 → 384
            in_memory:   If True, use local in-memory Qdrant (dev mode).
                         If False, connect to running Qdrant server.
        """
        self.collection_name = collection_name
        self.vector_size     = vector_size
        self.in_memory       = in_memory
        self.host            = host
        self.port            = port
        self._client         = None

        logger.info(f"Initializing QdrantVectorStore", extra={
            "collection":  collection_name,
            "vector_size": vector_size,
            "mode":        "in_memory" if in_memory else f"{host}:{port}",
        })

    @property
    def client(self):
        """Lazy-load the Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError("Run: pip install qdrant-client")

            if self.in_memory:
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(host=self.host, port=self.port)

            # Create collection if it doesn't exist
            self._ensure_collection()

        return self._client

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]

        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection", extra={
                "collection":  self.collection_name,
                "vector_size": self.vector_size,
            })

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

        from qdrant_client.models import PointStruct

        points = []
        for chunk, vector in zip(chunks, vectors):
            # Convert numpy array to list
            vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

            # Qdrant payload = metadata + content
            # We store content in payload so we can retrieve it without
            # keeping a separate mapping
            payload = {"content": chunk.content, **chunk.metadata}

            # Ensure all payload values are JSON-serializable
            clean_payload = {}
            for k, v in payload.items():
                if isinstance(v, (str, int, float, bool, list)):
                    clean_payload[k] = v
                else:
                    clean_payload[k] = str(v)

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec_list,
                payload=clean_payload,
            ))

        # Upload in batches for large collections
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Added chunks to Qdrant", extra={
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
        Search Qdrant using its HNSW index.
        Returns results with cosine similarity scores.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Build filter if provided
        qdrant_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        raw_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for rank, hit in enumerate(raw_results, start=1):
            payload = hit.payload or {}
            content = payload.pop("content", "")

            chunk = Chunk(content=content, metadata=payload)
            results.append(SearchResult(
                chunk=chunk,
                score=float(hit.score),
                rank=rank,
            ))

        logger.debug(f"Qdrant search complete", extra={
            "top_k":   top_k,
            "results": len(results),
        })

        return results

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._client = None
        logger.info(f"Deleted collection", extra={"collection": self.collection_name})

    def get_stats(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "collection": self.collection_name,
            "count":      info.points_count,
            "backend":    "qdrant",
            "vector_size": self.vector_size,
        }

    def collection_exists(self) -> bool:
        try:
            stats = self.get_stats()
            return stats["count"] > 0
        except Exception:
            return False