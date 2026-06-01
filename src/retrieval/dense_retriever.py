"""
src/retrieval/dense_retriever.py

Dense retriever — full pipeline from raw documents to answers
using dense vector search only.

This is the simplest complete RAG pipeline:
    documents → chunks → embeddings → vector store → retrieve → generate

Usage:
    from src.retrieval.dense_retriever import DenseRetriever
    retriever = DenseRetriever()
    retriever.index(documents)
    result = retriever.retrieve("What is attention mechanism?")
    print(result.answer)
"""

from dataclasses import dataclass, field
from src.data.chunker import Chunk, get_chunker
from src.data.loader import Document
from src.embeddings.dense import DenseEmbedder
from src.vectorstore.base import BaseVectorStore, SearchResult
from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Retrieval result container
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Complete result from the RAG pipeline.
    Contains both the answer and full transparency
    about how it was produced.
    """
    question:         str
    answer:           str
    source_chunks:    list[SearchResult]
    chunking_strategy: str
    retrieval_method:  str
    latency_ms:        float
    metadata:          dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(\n"
            f"  question='{self.question[:60]}...'\n"
            f"  answer='{self.answer[:80]}...'\n"
            f"  sources={len(self.source_chunks)}\n"
            f"  method={self.retrieval_method}\n"
            f"  latency={self.latency_ms:.0f}ms\n"
            f")"
        )


# ─────────────────────────────────────────────
# 2. Dense retriever
# ─────────────────────────────────────────────

class DenseRetriever:
    """
    Complete RAG pipeline using dense vector search.

    Indexing pipeline:
        documents → chunker → embedder → vector store

    Retrieval pipeline:
        question → embed → vector search → reformat → LLM → answer
    """

    def __init__(
        self,
        chunking_strategy: str             = "sliding_window",
        vector_store:      BaseVectorStore = None,
        top_k:             int             = settings.retrieval.top_k,
    ):
        self.chunking_strategy = chunking_strategy
        self.top_k             = top_k
        self.embedder          = DenseEmbedder()
        self.vector_store      = vector_store or ChromaVectorStore(
            collection_name=f"rag_{chunking_strategy}"
        )
        self.chunker = get_chunker(chunking_strategy)

        logger.info(f"Initialized DenseRetriever", extra={
            "strategy": chunking_strategy,
            "top_k":    top_k,
        })

    def index(self, documents: list[Document]) -> int:
        """
        Process documents and store in vector store.

        Args:
            documents: Raw loaded documents from DocumentLoader.

        Returns:
            Number of chunks indexed.
        """
        logger.info(f"Indexing documents", extra={
            "n_docs": len(documents)
        })

        # Step 1: chunk
        chunks = self.chunker.chunk(documents)
        logger.info(f"Chunking complete", extra={"n_chunks": len(chunks)})

        # Step 2: embed
        texts   = [chunk.content for chunk in chunks]
        vectors = self.embedder.embed(texts, show_progress=True)

        # Step 3: store
        self.vector_store.add_chunks(chunks, vectors)

        logger.info(f"Indexing complete", extra={
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
        Retrieve relevant chunks for a question.
        Returns raw SearchResults without generating an answer.
        Useful for evaluation.
        """
        query_vector = self.embedder.embed_single(question)
        results      = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k or self.top_k,
        )
        return results

    def retrieve(
        self,
        question: str,
        top_k:    int | None = None,
    ) -> RetrievalResult:
        """
        Full RAG pipeline: question → answer.

        Args:
            question: The user's question.
            top_k:    Override number of chunks to retrieve.

        Returns:
            RetrievalResult with answer and source chunks.
        """
        import time
        start_time = time.time()

        # Step 1: retrieve chunks
        source_chunks = self.retrieve_chunks(question, top_k=top_k)

        if not source_chunks:
            return RetrievalResult(
                question=question,
                answer="No relevant documents found in the knowledge base.",
                source_chunks=[],
                chunking_strategy=self.chunking_strategy,
                retrieval_method="dense",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: generate answer
        from src.generation.ollama_client import get_llm_client
        from src.generation.prompt_templates import RAGPromptTemplate

        chunks   = [r.chunk for r in source_chunks]
        template = RAGPromptTemplate()
        prompt   = template.format(question=question, chunks=chunks)

        client   = get_llm_client()
        response = client.generate_from_template(prompt)

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            question=question,
            answer=response.content,
            source_chunks=source_chunks,
            chunking_strategy=self.chunking_strategy,
            retrieval_method="dense",
            latency_ms=latency_ms,
            metadata={
                "prompt_tokens":  response.prompt_tokens,
                "output_tokens":  response.output_tokens,
                "llm_model":      response.model,
            }
        )