"""
Ask questions from your PDF files.
Usage: python notebooks/ask_pdf.py
"""

from src.data.loader import DocumentLoader
from src.retrieval.dense_retriever import DenseRetriever

# ── Load your PDF ──
loader   = DocumentLoader()
docs     = loader.load("src/data/raw/attention_is_all_you_need.pdf")  # change this to your file name
print(f"Loaded {len(docs)} pages")

# ── Index it ──
retriever = DenseRetriever(chunking_strategy="sliding_window")
n_chunks  = retriever.index(docs)
print(f"Indexed {n_chunks} chunks")

# ── Ask questions in a loop ──
print("\nReady! Type your questions (or 'quit' to exit)\n")

while True:
    question = input("Your question: ").strip()

    if question.lower() in ["quit", "exit", "q"]:
        break

    if not question:
        continue

    # Retrieve relevant chunks
    results = retriever.retrieve_chunks(question, top_k=3)

    print(f"\nTop {len(results)} relevant chunks:\n")
    for r in results:
        print(f"  Rank {r.rank} | Score {r.score:.4f} | Page {r.chunk.metadata.get('page', '?')}")
        print(f"  {r.chunk.content[:300]}")
        print()