"""
src/retrieval/hybrid_retriever.py

Full RAG pipeline using hybrid search (dense + sparse)
with optional cross-encoder reranking.

This is the most powerful retrieval configuration:
    1. Dense search     → semantic similarity
    2. Sparse BM25      → keyword matching
    3. RRF fusion       → combine both rankings
    4. Cross-encoder    → rerank top candidates precisely

Usage:
    from src.retrieval.hybrid_retriever import HybridRAGRetriever
    retriever = HybridRAGRetriever(use_reranker=True)
    retriever.index(documents)
    result = retriever.retrieve("What is attention mechanism?")
    print(result.answer)
"""

import time
from src.data.loader import Document
from src.data.chunker import get_chunker
from src.embeddings.hybrid import HybridRetriever
from src.retrieval.dense_retriever import RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker
from src.vectorstore.base import SearchResult
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRAGRetriever:
    """
    Production-grade RAG pipeline combining:
        - Hybrid search (dense + BM25 via RRF)
        - Optional cross-encoder reranking
        - Full answer generation

    This is what you'd deploy in a real system.
    The dense-only retriever is good for benchmarking.
    This one is good for production quality.
    """

    def __init__(
        self,
        chunking_strategy: str  = "sliding_window",
        use_reranker:      bool = True,
        top_k:             int  = settings.retrieval.top_k,
        fetch_k:           int  = 20,
    ):
        """
        Args:
            chunking_strategy: Which chunking strategy to use.
            use_reranker:      Whether to apply cross-encoder reranking.
            top_k:             Final number of chunks to return.
            fetch_k:           Candidates to fetch before reranking.
                               Should be > top_k. More = better recall,
                               slower reranking.
        """
        self.chunking_strategy = chunking_strategy
        self.use_reranker      = use_reranker
        self.top_k             = top_k
        self.fetch_k           = fetch_k

        self.chunker          = get_chunker(chunking_strategy)
        self.hybrid_retriever = HybridRetriever()
        self.reranker         = CrossEncoderReranker() if use_reranker else None

        logger.info(f"Initialized HybridRAGRetriever", extra={
            "strategy":     chunking_strategy,
            "use_reranker": use_reranker,
            "top_k":        top_k,
            "fetch_k":      fetch_k,
        })

    def index(self, documents: list[Document]) -> int:
        """
        Chunk documents and build hybrid index.

        Args:
            documents: Raw loaded documents.

        Returns:
            Number of chunks indexed.
        """
        logger.info(f"Indexing documents for hybrid retrieval", extra={
            "n_docs": len(documents)
        })

        chunks = self.chunker.chunk(documents)
        self.hybrid_retriever.fit(chunks)

        logger.info(f"Hybrid index built", extra={
            "n_chunks": len(chunks),
            "strategy": self.chunking_strategy,
        })

        return len(chunks)

    def retrieve_chunks(
        self,
        question: str,
        top_k:    int | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve and optionally rerank chunks for a question.
        Returns raw SearchResults — useful for evaluation.
        """
        # Step 1: hybrid retrieval — fetch more than top_k for reranking
        candidates = self.hybrid_retriever.retrieve(
            query=question,
            top_k=self.fetch_k,
        )

        # Convert (chunk, score) tuples to SearchResult objects
        search_results = [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(candidates, start=1)
        ]

        # Step 2: rerank if enabled
        if self.reranker and len(search_results) > 1:
            search_results = self.reranker.rerank(
                query=question,
                results=search_results,
                top_k=top_k or self.top_k,
            )
        else:
            search_results = search_results[:top_k or self.top_k]

        return search_results

    def retrieve(
        self,
        question: str,
        top_k:    int | None = None,
    ) -> RetrievalResult:
        """
        Full RAG pipeline: question → answer.
        Uses hybrid search + optional reranking + LLM generation.
        """
        start_time = time.time()

        # Retrieve chunks
        source_chunks = self.retrieve_chunks(question, top_k=top_k)

        if not source_chunks:
            return RetrievalResult(
                question=question,
                answer="No relevant documents found in the knowledge base.",
                source_chunks=[],
                chunking_strategy=self.chunking_strategy,
                retrieval_method="hybrid",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Generate answer
        from src.generation.ollama_client import get_llm_client
        from src.generation.prompt_templates import RAGPromptTemplate

        chunks   = [r.chunk for r in source_chunks]
        template = RAGPromptTemplate()
        prompt   = template.format(question=question, chunks=chunks)

        client   = get_llm_client()
        response = client.generate_from_template(prompt)

        latency_ms = (time.time() - start_time) * 1000

        method = "hybrid+reranker" if self.use_reranker else "hybrid"

        return RetrievalResult(
            question=question,
            answer=response.content,
            source_chunks=source_chunks,
            chunking_strategy=self.chunking_strategy,
            retrieval_method=method,
            latency_ms=latency_ms,
            metadata={
                "prompt_tokens": response.prompt_tokens,
                "output_tokens": response.output_tokens,
                "llm_model":     response.model,
            }
        )