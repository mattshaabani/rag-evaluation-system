"""
src/embeddings/sparse.py

Sparse retrieval using BM25 algorithm.
Keyword-based search that complements dense semantic search.

BM25 scores documents by term frequency weighted by how rare
each term is across the entire corpus (IDF).

Usage:
    from src.embeddings.sparse import BM25Retriever
    retriever = BM25Retriever()
    retriever.fit(chunks)               # build index
    results = retriever.retrieve("attention mechanism", top_k=5)
"""

import numpy as np
import re
from src.data.chunker import Chunk
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric.
    'Hello, World!' → ['hello', 'world']

    In production you'd use a proper tokenizer with stemming
    (e.g. 'running' and 'runs' both → 'run').
    For now this is good enough.
    """
    text   = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


class BM25Retriever:
    """
    BM25 sparse retrieval over a corpus of chunks.

    BM25 formula for term t in document d:
        score(t,d) = IDF(t) × tf_norm(t,d)

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        where:
            N     = total number of documents
            df(t) = number of documents containing term t

        tf_norm(t,d) = tf(t,d) × (k1 + 1)
                       ───────────────────────────────────────
                       tf(t,d) + k1 × (1 - b + b × |d|/avgdl)
        where:
            tf(t,d) = frequency of t in d
            k1      = saturation parameter (default 1.5)
            b       = length normalization (default 0.75)
            |d|     = length of document d
            avgdl   = average document length

    Parameters:
        k1: Controls term frequency saturation.
            High k1 → keeps rewarding repeated terms more
            Low k1  → quickly saturates (diminishing returns)
        b:  Controls length normalization.
            b=0 → no length normalization
            b=1 → full normalization
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1     = k1
        self.b      = b
        self.chunks = []
        self.corpus_tokens: list[list[str]] = []

        # These are computed at fit() time
        self.df: dict[str, int]      = {}   # document frequency per term
        self.idf: dict[str, float]   = {}   # IDF score per term
        self.avgdl: float            = 0.0  # average document length
        self.n_docs: int             = 0    # total number of documents

    def fit(self, chunks: list[Chunk]) -> "BM25Retriever":
        """
        Build the BM25 index from a list of chunks.
        Must be called before retrieve().

        This computes:
        - Token list for each chunk
        - Document frequency (df) for each term
        - IDF score for each term
        - Average document length
        """
        logger.info(f"Building BM25 index", extra={"n_chunks": len(chunks)})

        self.chunks        = chunks
        self.corpus_tokens = [tokenize(chunk.content) for chunk in chunks]
        self.n_docs        = len(chunks)

        # Compute average document length
        doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl  = np.mean(doc_lengths) if doc_lengths else 1.0

        # Compute document frequency for each term
        self.df = {}
        for tokens in self.corpus_tokens:
            for term in set(tokens):   # set() → count each term once per doc
                self.df[term] = self.df.get(term, 0) + 1

        # Compute IDF for each term
        # Formula: log((N - df + 0.5) / (df + 0.5) + 1)
        # The +1 inside log prevents negative IDF for very common terms
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = np.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )

        logger.info(f"BM25 index built", extra={
            "vocab_size": len(self.df),
            "avg_doc_length": round(self.avgdl, 1),
        })

        return self

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """
        Compute BM25 score for one document given query tokens.
        Sum BM25(t,d) across all query terms.
        """
        doc_tokens = self.corpus_tokens[doc_idx]
        doc_len    = len(doc_tokens)

        # Term frequency map for this document
        tf_map: dict[str, int] = {}
        for token in doc_tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue   # term not in corpus vocabulary → skip

            tf  = tf_map.get(term, 0)
            idf = self.idf[term]

            # BM25 term frequency normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )

            score += idf * tf_norm

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval.top_k,
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Returns:
            List of (chunk, score) tuples sorted by score descending.
        """
        if not self.chunks:
            raise RuntimeError("BM25Retriever not fitted. Call fit() first.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Score all documents
        scores = [
            self._score_document(query_tokens, i)
            for i in range(self.n_docs)
        ]

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.chunks[i], scores[i])
            for i in top_indices
            if scores[i] > 0   # only return chunks with non-zero score
        ]

        logger.debug(f"BM25 retrieval complete", extra={
            "query":    query[:50],
            "top_k":    top_k,
            "results":  len(results),
        })

        return results