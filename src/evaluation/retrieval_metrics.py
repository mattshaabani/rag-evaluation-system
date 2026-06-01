"""
src/evaluation/retrieval_metrics.py

Information retrieval metrics for evaluating chunk retrieval quality.
These measure HOW WELL we retrieve relevant chunks — independent of LLM.

Metrics implemented:
    Precision@K  — of top K retrieved, what fraction are relevant?
    Recall@K     — of all relevant chunks, what fraction did we retrieve?
    MRR          — where does the first relevant chunk appear?
    NDCG@K       — position-weighted relevance score

All metrics take:
    retrieved:  list of chunk IDs we retrieved (ranked)
    relevant:   set of chunk IDs that are actually relevant (ground truth)

Usage:
    from src.evaluation.retrieval_metrics import compute_all_metrics
    metrics = compute_all_metrics(retrieved=['c1','c2','c3'], relevant={'c1','c3'}, k=5)
"""

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Individual metrics
# ─────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K — of the top K retrieved chunks, what fraction are relevant?

    Formula:
        P@K = |retrieved[:K] ∩ relevant| / K

    Example:
        retrieved = [A, B, C, D, E]
        relevant  = {A, C, E}
        P@3 = |{A,B,C} ∩ {A,C,E}| / 3 = 2/3 = 0.667

    Range: 0 to 1. Higher is better.
    """
    if k == 0:
        return 0.0
    top_k     = retrieved[:k]
    n_relevant = sum(1 for doc in top_k if doc in relevant)
    return n_relevant / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K — of ALL relevant chunks, what fraction did we retrieve in top K?

    Formula:
        R@K = |retrieved[:K] ∩ relevant| / |relevant|

    Example:
        retrieved = [A, B, C, D, E]
        relevant  = {A, C, F}         ← F not retrieved at all
        R@3 = |{A,B,C} ∩ {A,C,F}| / 3 = 2/3 = 0.667

    Range: 0 to 1. Higher is better.
    Note: if |relevant| = 0, returns 0.
    """
    if not relevant:
        return 0.0
    top_k      = retrieved[:k]
    n_relevant = sum(1 for doc in top_k if doc in relevant)
    return n_relevant / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal Rank — where does the FIRST relevant chunk appear?

    Formula:
        RR = 1 / rank_of_first_relevant

    Example:
        retrieved = [B, C, A, D, E]
        relevant  = {A, E}
        First relevant is A at rank 3 → RR = 1/3 = 0.333

    Range: 0 to 1. Higher is better.
    1.0 = first result is relevant
    0.5 = second result is relevant
    0.0 = no relevant result found
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain.
    The most comprehensive retrieval metric.

    Key insight: a relevant chunk at rank 1 is worth more than
    the same chunk at rank 5. NDCG captures this with a
    logarithmic discount.

    Formula:
        DCG@K  = Σ rel_i / log2(i + 1)    for i = 1..K
        IDCG@K = DCG of perfect ranking    (all relevant docs first)
        NDCG@K = DCG@K / IDCG@K

    Where rel_i = 1 if chunk at rank i is relevant, 0 otherwise.

    Example:
        retrieved = [A, B, C, D, E]
        relevant  = {A, C}
        DCG@5  = 1/log2(2) + 0 + 1/log2(4) + 0 + 0
               = 1.0 + 0.5 = 1.5
        IDCG@5 = 1/log2(2) + 1/log2(3)     ← perfect: A,C first
               = 1.0 + 0.631 = 1.631
        NDCG@5 = 1.5 / 1.631 = 0.920

    Range: 0 to 1. Higher is better.
    """
    if not relevant or k == 0:
        return 0.0

    # Compute DCG for actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 1)

    # Compute ideal DCG (perfect ranking — all relevant docs first)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(
        1.0 / np.log2(i + 1)
        for i in range(1, n_relevant_in_k + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ─────────────────────────────────────────────
# 2. Compute all metrics at once
# ─────────────────────────────────────────────

def compute_all_metrics(
    retrieved: list[str],
    relevant:  set[str],
    k_values:  list[int] = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for one query.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank 1 first).
        relevant:  Set of chunk IDs that are ground-truth relevant.
        k_values:  List of K values to compute metrics at.

    Returns:
        Dict of metric_name → score. Example:
        {
            "precision@1": 1.0,
            "precision@3": 0.67,
            "precision@5": 0.4,
            "recall@5":    0.8,
            "mrr":         1.0,
            "ndcg@5":      0.92,
        }
    """
    from src.utils.config import settings
    k_values = k_values or settings.evaluation.k_values

    metrics = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"]    = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"]      = ndcg_at_k(retrieved, relevant, k)

    metrics["mrr"] = reciprocal_rank(retrieved, relevant)

    return metrics


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Average metrics across multiple queries.
    This gives you dataset-level scores (MAP, MNDCG etc.)

    Args:
        all_metrics: List of per-query metric dicts.

    Returns:
        Averaged metrics dict.
    """
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    return {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in keys
    }