"""
Run the full A/B test comparing all three chunking strategies.
Results are logged to MLflow and printed as a comparison table.

Run with:
    python notebooks/run_ab_test.py
"""

import json
from src.data.loader import DocumentLoader
from src.evaluation.ab_test import ChunkingStrategyABTest
from src.evaluation.report import EvaluationReport


def main():
    # Load documents
    print("Loading documents...")
    loader = DocumentLoader()
    docs   = loader.load("src/data/raw/rag_overview.txt")
    print(f"Loaded {len(docs)} document(s)")

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    with open("src/data/eval_datasets/qa_pairs.json") as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Run A/B test
    print("\nRunning A/B test across all chunking strategies...")
    ab_test = ChunkingStrategyABTest()
    results = ab_test.run(
        documents=docs,
        qa_pairs=qa_pairs,
    )

    # Generate and print report
    report = EvaluationReport(results)
    report.print_summary()
    report.save("src/data/eval_datasets/final_report.json")

    print("\nMLflow UI: run 'mlflow ui' then open http://localhost:5000")


if __name__ == "__main__":
    main()