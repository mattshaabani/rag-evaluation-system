"""
src/embeddings/hybrid.py

Hybrid retrieval: combines dense (semantic) + sparse (BM25) search
using Reciprocal Rank Fusion (RRF).

Why hybrid?
    Dense search finds semantically similar chunks even with
    different wording. Sparse search finds exact keyword matches.
    Together they cover each other's blind spots.

RRF formula:
    RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                    systems

    where rank_i(doc) is the document's rank in system i (1-indexed)
    and k=60 is a smoothing constant that reduces the impact of
    very high ranks.

Usage:
    from src.embeddings.hybrid import HybridRetriever
    retriever = HybridRetriever()
    retriever.fit(chunks)
    results = retriever.retrieve("attention mechanism", top_k=5)
"""

import numpy as np
from src.data.chunker import Chunk
from src.embeddings.dense import DenseEmbedder
from src.embeddings.sparse import BM25Retriever
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines dense vector search and BM25 sparse search via RRF.

    RRF intuition:
        Imagine two ranked lists from two different search systems:
            Dense:  [chunk_A, chunk_C, chunk_B, chunk_D]
            Sparse: [chunk_C, chunk_A, chunk_D, chunk_B]

        chunk_A: dense_rank=1, sparse_rank=2
            RRF = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325

        chunk_C: dense_rank=2, sparse_rank=1
            RRF = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

        chunk_B: dense_rank=3, sparse_rank=4
            RRF = 1/(60+3) + 1/(60+4) = 0.0159 + 0.0156 = 0.0315

        Final ranking: [chunk_A ≈ chunk_C, chunk_B, chunk_D]
        Documents that rank well in BOTH systems float to the top.
    """

    def __init__(
        self,
        dense_weight:  float = settings.embedding.hybrid_weight,
        rrf_k:         int   = 60,
        model_name:    str   = settings.embedding.dense_model,
    ):
        """
        Args:
            dense_weight: Weight for dense results in final fusion.
                          0.0 = pure sparse, 1.0 = pure dense, 0.5 = equal.
                          Note: RRF naturally balances them, this is an
                          additional tuning knob.
            rrf_k:        RRF smoothing constant. Higher k = less penalty
                          for lower ranks. 60 is the standard default.
        """
        self.dense_weight  = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.rrf_k         = rrf_k

        self.dense_embedder  = DenseEmbedder(model_name=model_name)
        self.sparse_retriever = BM25Retriever()

        self.chunks: list[Chunk]    = []
        self.chunk_vectors: np.ndarray | None = None

    def fit(self, chunks: list[Chunk]) -> "HybridRetriever":
        """
        Build both the dense index and the BM25 index.

        Dense index:  embed all chunks → store matrix of vectors
        Sparse index: build BM25 term frequencies and IDF scores
        """
        logger.info(f"Building hybrid index", extra={"n_chunks": len(chunks)})

        self.chunks = chunks

        # Build sparse index
        self.sparse_retriever.fit(chunks)

        # Build dense index — embed all chunks at once
        texts              = [chunk.content for chunk in chunks]
        self.chunk_vectors = self.dense_embedder.embed(
            texts,
            show_progress=True
        )

        logger.info(f"Hybrid index built", extra={
            "n_chunks":       len(chunks),
            "embedding_dim":  self.chunk_vectors.shape[1],
        })

        return self

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """
        Dense retrieval: embed query, compute dot product with all chunks.
        Returns list of (chunk_index, similarity_score).

        Why dot product instead of cosine similarity?
        Because we normalize embeddings at encode time (normalize_embeddings=True
        in DenseEmbedder), so ||vec|| = 1 for all vectors.
        cosine_sim(A,B) = dot(A,B) / (||A|| × ||B||) = dot(A,B) / (1 × 1) = dot(A,B)
        Dot product on normalized vectors IS cosine similarity.
        """
        query_vec = self.dense_embedder.embed_single(query)

        # Matrix multiplication: (n_chunks, dim) × (dim,) → (n_chunks,)
        # This computes similarity of query with ALL chunks simultaneously
        similarities = self.chunk_vectors @ query_vec

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _rrf_score(
        self,
        dense_results:  list[tuple[int, float]],
        sparse_results: list[tuple[Chunk, float]],
    ) -> dict[int, float]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.

        Args:
            dense_results:  [(chunk_index, score), ...] from dense search
            sparse_results: [(chunk, score), ...]       from BM25 search

        Returns:
            dict mapping chunk_index → RRF score
        """
        rrf_scores: dict[int, float] = {}

        # Process dense rankings
        for rank, (chunk_idx, _) in enumerate(dense_results, start=1):
            rrf_contribution = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + rrf_contribution

        # Build reverse lookup: chunk object → index in self.chunks
        chunk_to_idx = {id(chunk): idx for idx, chunk in enumerate(self.chunks)}

        # Process sparse rankings
        for rank, (chunk, _) in enumerate(sparse_results, start=1):
            chunk_idx = chunk_to_idx.get(id(chunk))
            if chunk_idx is None:
                continue
            rrf_contribution = self.sparse_weight / (self.rrf_k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + rrf_contribution

        return rrf_scores

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval.top_k,
        fetch_k: int = 20,
    ) -> list[tuple[Chunk, float]]:
        """
        Hybrid retrieval using RRF fusion.

        Args:
            query:   The search query string.
            top_k:   Number of results to return after fusion.
            fetch_k: How many results to fetch from each system
                     before fusion. Should be > top_k.

        Returns:
            List of (chunk, rrf_score) sorted by score descending.
        """
        if not self.chunks:
            raise RuntimeError("HybridRetriever not fitted. Call fit() first.")

        # Fetch from both systems
        dense_results  = self._dense_retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k)

        # Fuse with RRF
        rrf_scores = self._rrf_score(dense_results, sparse_results)

        # Sort by RRF score and return top_k
        sorted_indices = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        top_indices    = sorted_indices[:top_k]

        results = [
            (self.chunks[idx], rrf_scores[idx])
            for idx in top_indices
        ]

        logger.debug(f"Hybrid retrieval complete", extra={
            "query":   query[:50],
            "top_k":   top_k,
            "results": len(results),
        })

        return results