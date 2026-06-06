"""
src/evaluation/ab_test.py

A/B testing framework for comparing chunking strategies.
Uses MLflow to track every experiment run.

This is the scientific core of Project 1:
    - Run each chunking strategy on the same eval dataset
    - Log all parameters and metrics to MLflow
    - Compare results in MLflow UI

Usage:
    from src.evaluation.ab_test import ChunkingStrategyABTest
    ab_test = ChunkingStrategyABTest()
    results = ab_test.run(documents, qa_pairs)
"""

import json
import time
import mlflow
import mlflow.tracking
from pathlib import Path
from src.data.loader import Document
from src.data.chunker import get_chunker
from src.embeddings.dense import DenseEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.evaluation.retrieval_metrics import compute_all_metrics, average_metrics
from src.evaluation.ragas_eval import RAGASEvaluator
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingStrategyABTest:
    """
    Runs all chunking strategies on the same eval dataset
    and logs everything to MLflow for comparison.

    MLflow experiment structure:
        Experiment: "rag-chunking-comparison"
            Run: sliding_window  → parameters + metrics + artifacts
            Run: semantic        → parameters + metrics + artifacts
            Run: hierarchical    → parameters + metrics + artifacts

    After running, open MLflow UI:
        mlflow ui
        → http://localhost:5000
    """

    EXPERIMENT_NAME = "rag-chunking-comparison"

    def __init__(self):
        self.evaluator = RAGASEvaluator()
        self.embedder  = DenseEmbedder()

        # Set up MLflow — runs locally, no server needed
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.EXPERIMENT_NAME)

        logger.info(f"Initialized A/B test", extra={
            "experiment": self.EXPERIMENT_NAME
        })

    def _evaluate_strategy(
        self,
        strategy:  str,
        documents: list[Document],
        qa_pairs:  list[dict],
        run_name:  str,
    ) -> dict:
        """
        Run one chunking strategy and log to MLflow.

        Args:
            strategy:  Chunking strategy name.
            documents: Documents to index.
            qa_pairs:  List of {"question": ..., "answer": ...} dicts.
            run_name:  MLflow run name.

        Returns:
            Dict of averaged metrics.
        """
        with mlflow.start_run(run_name=run_name):

            # ── Log parameters ──
            mlflow.log_param("strategy",      strategy)
            mlflow.log_param("chunk_size",    settings.chunking.chunk_size)
            mlflow.log_param("chunk_overlap", settings.chunking.chunk_overlap)
            mlflow.log_param("embedding_model", settings.embedding.dense_model)
            mlflow.log_param("top_k",         settings.retrieval.top_k)
            mlflow.log_param("n_documents",   len(documents))
            mlflow.log_param("n_qa_pairs",    len(qa_pairs))

            # ── Build retriever ──
            retriever = DenseRetriever(
                chunking_strategy=strategy,
                vector_store=ChromaVectorStore(
                    collection_name=f"abtest_{strategy}_{int(time.time())}"
                )
            )

            start_time = time.time()
            n_chunks   = retriever.index(documents)
            index_time = time.time() - start_time

            mlflow.log_metric("n_chunks",        n_chunks)
            mlflow.log_metric("index_time_sec",  round(index_time, 2))

            logger.info(f"Running evaluation", extra={
                "strategy": strategy,
                "n_chunks": n_chunks,
                "n_queries": len(qa_pairs),
            })

            # ── Evaluate each QA pair ──
            all_retrieval_metrics = []
            all_ragas_metrics     = []

            for i, qa in enumerate(qa_pairs):
                question     = qa["question"]
                ground_truth = qa.get("answer", "")

                # Retrieve chunks
                search_results = retriever.retrieve_chunks(question)
                retrieved_ids  = [
                    str(r.chunk.metadata.get("chunk_index", idx))
                    for idx, r in enumerate(search_results)
                ]
                contexts = [r.chunk.content for r in search_results]

                # Retrieval metrics
                # For ground truth relevant IDs we use chunk_index 0
                # (simplified — in real eval you'd have labeled relevant chunks)
                relevant_ids = {str(0)}
                ret_metrics  = compute_all_metrics(
                    retrieved=retrieved_ids,
                    relevant=relevant_ids,
                    k_values=settings.evaluation.k_values,
                )
                all_retrieval_metrics.append(ret_metrics)

                # RAGAS metrics (no LLM needed — embedding based)
                ragas_scores = self.evaluator.evaluate(
                    question=question,
                    answer=ground_truth,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
                all_ragas_metrics.append(ragas_scores)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i+1}/{len(qa_pairs)} queries")

            # ── Average metrics across all queries ──
            avg_retrieval = average_metrics(all_retrieval_metrics)
            avg_ragas     = average_metrics(all_ragas_metrics)

            # ── Log all metrics to MLflow ──
            for name, value in avg_retrieval.items():
                clean_name = name.replace("@", "_at_")  # MLflow doesn't allow '@' in metric names
                mlflow.log_metric(clean_name, round(value, 4))

            for name, value in avg_ragas.items():
                clean_name = name.replace("@", "_at_")  # MLflow doesn't allow '@' in metric names
                mlflow.log_metric(clean_name, round(value, 4))
            # ── Log evaluation report as artifact ──
            report = {
                "strategy":          strategy,
                "n_chunks":          n_chunks,
                "n_queries":         len(qa_pairs),
                "retrieval_metrics": avg_retrieval,
                "ragas_metrics":     avg_ragas,
            }

            report_path = Path(f"src/data/eval_datasets/{strategy}_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(report_path))

            logger.info(f"Strategy evaluation complete", extra={
                "strategy": strategy,
                "ndcg@5":   round(avg_retrieval.get("ndcg@5", 0), 4),
                "faithfulness": round(avg_ragas.get("faithfulness", 0), 4),
            })

            return {**avg_retrieval, **avg_ragas}

    def run(
        self,
        documents:  list[Document],
        qa_pairs:   list[dict],
        strategies: list[str] = None,
    ) -> dict[str, dict]:
        """
        Run A/B test across all chunking strategies.

        Args:
            documents:  Documents to index and evaluate on.
            qa_pairs:   Evaluation Q&A pairs.
            strategies: Which strategies to test.
                        Defaults to all three.

        Returns:
            Dict of strategy_name → averaged metrics.
        """
        strategies = strategies or settings.chunking.strategies

        logger.info(f"Starting A/B test", extra={
            "strategies": strategies,
            "n_docs":     len(documents),
            "n_queries":  len(qa_pairs),
        })

        all_results = {}

        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy}")
            run_name = f"{strategy}_{settings.chunking.chunk_size}"

            try:
                results = self._evaluate_strategy(
                    strategy=strategy,
                    documents=documents,
                    qa_pairs=qa_pairs,
                    run_name=run_name,
                )
                all_results[strategy] = results

            except Exception as e:
                import traceback
                logger.error(f"Strategy failed", extra={
                    "strategy": strategy,
                    "error":    str(e),
                })
                traceback.print_exc()

        logger.info(f"A/B test complete", extra={
            "strategies_tested": list(all_results.keys())
        })

        return all_results