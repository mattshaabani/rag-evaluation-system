"""
src/retrieval/reranker.py

Cross-encoder reranker for improving retrieval precision.

Two-stage retrieval:
    Stage 1 — bi-encoder vector search (fast, retrieve top 20)
    Stage 2 — cross-encoder reranking (accurate, rerank to top 5)

Why cross-encoders are more accurate:
    Bi-encoder: embed(query) vs embed(chunk) separately
        → fast but loses interaction between query and chunk

    Cross-encoder: encode(query + chunk) together
        → sees full interaction, much more accurate
        → too slow for full corpus, perfect for reranking 20 candidates

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Trained on MS MARCO passage ranking dataset
    384 dimensions, runs locally, no API needed
"""

import numpy as np
from src.vectorstore.base import SearchResult
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    The cross-encoder takes (query, chunk) pairs and outputs
    a relevance score for each pair directly — not embeddings.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, search_results, top_k=5)
    """

    def __init__(
        self,
        model_name: str = settings.retrieval.reranker_model,
    ):
        self.model_name = model_name
        self._model     = None

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder reranker", extra={
                "model": self.model_name
            })
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query:   str,
        results: list[SearchResult],
        top_k:   int = settings.retrieval.top_k,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder scores.

        Args:
            query:   The original search query.
            results: Initial retrieval results to rerank.
            top_k:   Number of results to return after reranking.

        Returns:
            Reranked list of SearchResult with updated scores and ranks.
        """
        if not results:
            return []

        if len(results) <= 1:
            return results[:top_k]

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, result.chunk.content) for result in results]

        logger.debug(f"Reranking {len(pairs)} candidates")

        # Cross-encoder scores each pair jointly
        scores = self.model.predict(pairs)

        # Combine original results with new scores
        scored_results = list(zip(results, scores))

        # Sort by cross-encoder score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Rebuild SearchResult list with updated ranks and scores
        reranked = []
        for new_rank, (result, new_score) in enumerate(
            scored_results[:top_k], start=1
        ):
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(new_score),
                rank=new_rank,
            ))

        logger.debug(f"Reranking complete", extra={
            "input_count":  len(results),
            "output_count": len(reranked),
        })

        return reranked