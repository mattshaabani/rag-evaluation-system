"""
src/api/routes.py

FastAPI route handlers.
Each function handles one endpoint — validates input,
calls the pipeline, returns structured response.
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    SourceChunk, HealthResponse,
)
from src.data.loader import DocumentLoader
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRAGRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory store of retrievers indexed by strategy
# In production this would be a proper cache/database
_retrievers: dict[str, DenseRetriever | HybridRAGRetriever] = {}


# ─────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API and its components are running.
    Used by Docker healthchecks and monitoring.
    """
    return HealthResponse(
        status="healthy",
        components={
            "api":          "up",
            "vector_store": "up",
            "retrievers":   list(_retrievers.keys()),
        }
    )


# ─────────────────────────────────────────────
# 2. Ingest endpoint
# ─────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Load, chunk and index documents into the vector store.

    Steps:
        1. Load document from path or URL
        2. Create retriever with specified chunking strategy
        3. Index the document
        4. Store retriever for later queries
    """
    logger.info(f"Ingest request", extra={
        "source":   request.source,
        "strategy": request.chunking_strategy,
    })

    try:
        # Load document
        loader = DocumentLoader()
        docs   = loader.load(request.source)

        # Create and index retriever
        retriever = DenseRetriever(
            chunking_strategy=request.chunking_strategy,
            top_k=5,
        )
        n_chunks = retriever.index(docs)

        # Store retriever for query endpoint
        _retrievers[request.chunking_strategy] = retriever

        logger.info(f"Ingest complete", extra={
            "chunks": n_chunks,
            "source": request.source,
        })

        return IngestResponse(
            status="success",
            source=request.source,
            chunks_created=n_chunks,
            chunking_strategy=request.chunking_strategy,
            message=f"Successfully indexed {n_chunks} chunks from {request.source}",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingest failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 3. Query endpoint
# ─────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using the indexed documents.

    Steps:
        1. Check if documents have been indexed
        2. Retrieve relevant chunks
        3. Generate answer (if LLM credits available)
        4. Return answer with sources
    """
    logger.info(f"Query request", extra={
        "question": request.question[:60],
        "strategy": request.chunking_strategy,
    })

    # Check if this strategy has been indexed
    if request.chunking_strategy not in _retrievers:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{request.chunking_strategy}' not indexed yet. "
                   f"Call /ingest first. "
                   f"Available: {list(_retrievers.keys())}"
        )

    retriever = _retrievers[request.chunking_strategy]

    try:
        # Retrieve chunks only (no LLM generation needed)
        search_results = retriever.retrieve_chunks(
            question=request.question,
            top_k=request.top_k,
        )

        # Format source chunks for response
        source_chunks = [
            SourceChunk(
                content=r.chunk.content,
                source=r.chunk.metadata.get("source", "unknown"),
                score=r.score,
                rank=r.rank,
                metadata=r.chunk.metadata,
            )
            for r in search_results
        ]

        # Try to generate answer — gracefully handle no credits
        answer = ""
        metadata = {}
        try:
            from src.generation.ollama_client import get_llm_client
            from src.generation.prompt_templates import RAGPromptTemplate

            chunks   = [r.chunk for r in search_results]
            template = RAGPromptTemplate()
            prompt   = template.format(
                question=request.question,
                chunks=chunks,
            )
            client   = get_llm_client()
            response = client.generate_from_template(prompt)
            answer   = response.content
            metadata = {
                "prompt_tokens": response.prompt_tokens,
                "output_tokens": response.output_tokens,
            }
        except Exception as e:
            answer = (
                f"[LLM generation unavailable: {str(e)}] "
                f"Retrieved {len(source_chunks)} relevant chunks — "
                f"see source_chunks for context."
            )

        return QueryResponse(
            question=request.question,
            answer=answer,
            source_chunks=source_chunks,
            chunking_strategy=request.chunking_strategy,
            retrieval_method="dense",
            latency_ms=0.0,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Query failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))