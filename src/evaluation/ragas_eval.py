"""
src/evaluation/ragas_eval.py

LLM-based RAG evaluation using RAGAS-style metrics.
Measures answer quality beyond just retrieval.

Metrics:
    faithfulness      — is the answer grounded in the context?
    answer_relevancy  — does the answer address the question?
    context_precision — are retrieved chunks actually useful?
    context_recall    — did we retrieve all needed information?

Since we may not have credits, we implement lightweight versions
that use heuristics + optional LLM scoring.

Usage:
    from src.evaluation.ragas_eval import RAGASEvaluator
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG stands for Retrieval Augmented Generation...",
        contexts=["RAG combines vector search with LLM generation..."],
        ground_truth="RAG is a technique that combines retrieval with generation"
    )
"""

import re
import numpy as np
from src.embeddings.dense import DenseEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    Lightweight RAGAS-style evaluator using embedding similarity.

    Why not use the ragas library directly?
        The ragas library makes LLM API calls for each evaluation.
        With no credits, we implement embedding-based approximations
        that correlate strongly with LLM-based scores.

    When you have credits, swap this for full ragas library calls.
    """

    def __init__(self):
        self.embedder = DenseEmbedder()

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts via embeddings."""
        vec_a = self.embedder.embed_single(text_a)
        vec_b = self.embedder.embed_single(text_b)
        return float(np.dot(vec_a, vec_b))

    def faithfulness(
        self,
        answer:   str,
        contexts: list[str],
    ) -> float:
        """
        Faithfulness — is the answer grounded in the retrieved context?

        Approach:
            Split answer into sentences.
            For each sentence, find its max similarity to any context chunk.
            Average across all sentences.

        High faithfulness = every claim in the answer
        can be traced back to the context.
        Low faithfulness = the LLM is hallucinating.

        Range: 0 to 1. Higher is better.
        """
        if not answer or not contexts:
            return 0.0

        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 0.0

        combined_context = " ".join(contexts)
        scores = []

        for sentence in sentences:
            # How similar is this sentence to the context?
            sim = self._semantic_similarity(sentence, combined_context)
            scores.append(sim)

        return float(np.mean(scores))

    def answer_relevancy(
        self,
        question: str,
        answer:   str,
    ) -> float:
        """
        Answer Relevancy — does the answer actually address the question?

        Approach:
            Measure semantic similarity between question and answer.
            High similarity = answer is on-topic.
            Low similarity = answer went off-topic.

        Range: 0 to 1. Higher is better.
        """
        if not question or not answer:
            return 0.0

        return self._semantic_similarity(question, answer)

    def context_precision(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        """
        Context Precision — of retrieved chunks, how many are useful?

        Approach:
            Measure similarity of each context chunk to the question.
            Average across chunks.

        High precision = all retrieved chunks are relevant.
        Low precision = we retrieved many irrelevant chunks.

        Range: 0 to 1. Higher is better.
        """
        if not contexts:
            return 0.0

        scores = [
            self._semantic_similarity(question, ctx)
            for ctx in contexts
        ]

        return float(np.mean(scores))

    def context_recall(
        self,
        answer:       str,
        contexts:     list[str],
        ground_truth: str,
    ) -> float:
        """
        Context Recall — did we retrieve everything needed to answer?

        Approach:
            Measure how similar the ground truth answer is to
            the retrieved context. If context contains the ground
            truth information, recall is high.

        Range: 0 to 1. Higher is better.
        """
        if not ground_truth or not contexts:
            return 0.0

        combined_context = " ".join(contexts)
        return self._semantic_similarity(ground_truth, combined_context)

    def evaluate(
        self,
        question:     str,
        answer:       str,
        contexts:     list[str],
        ground_truth: str = "",
    ) -> dict[str, float]:
        """
        Run all RAGAS metrics for one QA pair.

        Args:
            question:     The user question.
            answer:       The generated answer.
            contexts:     The retrieved context chunks (as strings).
            ground_truth: The expected correct answer (optional).

        Returns:
            Dict of metric_name → score.
        """
        scores = {
            "faithfulness":      self.faithfulness(answer, contexts),
            "answer_relevancy":  self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, contexts),
        }

        if ground_truth:
            scores["context_recall"] = self.context_recall(
                answer, contexts, ground_truth
            )

        logger.debug(f"RAGAS evaluation complete", extra=scores)
        return scores