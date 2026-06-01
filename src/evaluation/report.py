"""
src/evaluation/report.py

Generates human-readable evaluation reports from A/B test results.
Prints a comparison table and saves a JSON summary.

Usage:
    from src.evaluation.report import EvaluationReport
    report = EvaluationReport(results)
    report.print_summary()
    report.save("data/eval_datasets/final_report.json")
"""

import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReport:
    """
    Formats and displays A/B test results.

    Takes the output of ChunkingStrategyABTest.run() and:
    - Prints a comparison table to terminal
    - Identifies the winning strategy per metric
    - Saves full report to JSON
    """

    def __init__(self, results: dict[str, dict]):
        """
        Args:
            results: Output of ChunkingStrategyABTest.run()
                     {strategy_name: {metric_name: score}}
        """
        self.results    = results
        self.strategies = list(results.keys())

    def _get_winner(self, metric: str) -> str:
        """Find which strategy scored highest on a metric."""
        best_strategy = max(
            self.strategies,
            key=lambda s: self.results[s].get(metric, 0)
        )
        return best_strategy

    def print_summary(self) -> None:
        """Print a formatted comparison table to terminal."""

        key_metrics = [
            "ndcg@5",
            "mrr",
            "precision@5",
            "recall@5",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ]

        # Header
        print("\n" + "="*70)
        print("RAG CHUNKING STRATEGY COMPARISON")
        print("="*70)

        # Column widths
        col_width = 18
        print(f"{'Metric':<22}", end="")
        for strategy in self.strategies:
            print(f"{strategy:>{col_width}}", end="")
        print(f"{'Winner':>{col_width}}")
        print("-"*70)

        # Rows
        for metric in key_metrics:
            if not any(metric in self.results[s] for s in self.strategies):
                continue

            print(f"{metric:<22}", end="")
            scores = {}
            for strategy in self.strategies:
                score = self.results[strategy].get(metric, 0)
                scores[strategy] = score
                print(f"{score:>{col_width}.4f}", end="")

            winner = max(scores, key=scores.get)
            print(f"{winner:>{col_width}}")

        print("="*70)

        # Overall recommendation
        win_counts = {s: 0 for s in self.strategies}
        for metric in key_metrics:
            if any(metric in self.results[s] for s in self.strategies):
                winner = self._get_winner(metric)
                win_counts[winner] += 1

        best_overall = max(win_counts, key=win_counts.get)
        print(f"\n✓ RECOMMENDED STRATEGY: {best_overall}")
        print(f"  (Won {win_counts[best_overall]}/{len(key_metrics)} metrics)")
        print("="*70 + "\n")

    def save(self, path: str = "data/eval_datasets/final_report.json") -> None:
        """Save full report to JSON file."""
        output = {
            "strategies":   self.strategies,
            "results":      self.results,
            "winners":      {
                metric: self._get_winner(metric)
                for metric in ["ndcg@5", "mrr", "faithfulness", "answer_relevancy"]
                if any(metric in self.results[s] for s in self.strategies)
            }
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2))

        logger.info(f"Report saved", extra={"path": str(path)})
        print(f"Full report saved to: {path}")