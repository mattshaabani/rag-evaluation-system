"""
src/api/schemas.py

Pydantic schemas for FastAPI request and response validation.
These define the exact shape of every API input and output.

Benefits:
    - Automatic input validation
    - Auto-generated OpenAPI docs at /docs
    - Type safety throughout the API layer
    - Clear error messages for bad requests
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────
# 1. Ingest endpoint schemas
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Request body for POST /ingest.
    Tells the system where to find documents and how to process them.
    """
    source:            str   = Field(
        ...,
        description="File path or URL to ingest",
        examples=["data/raw/paper.pdf", "https://example.com/article"]
    )
    chunking_strategy: str   = Field(
        default="sliding_window",
        description="Chunking strategy to use",
        examples=["sliding_window", "semantic", "hierarchical"]
    )
    chunk_size:        Optional[int] = Field(
        default=None,
        description="Override default chunk size"
    )
    chunk_overlap:     Optional[int] = Field(
        default=None,
        description="Override default chunk overlap"
    )


class IngestResponse(BaseModel):
    """Response from POST /ingest."""
    status:            str
    source:            str
    chunks_created:    int
    chunking_strategy: str
    message:           str


# ─────────────────────────────────────────────
# 2. Query endpoint schemas
# ─────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A single retrieved source chunk returned with the answer."""
    content:  str
    source:   str
    score:    float
    rank:     int
    metadata: dict


class QueryRequest(BaseModel):
    """
    Request body for POST /query.
    """
    question:          str  = Field(
        ...,
        description="The question to answer",
        examples=["What is the attention mechanism?"]
    )
    top_k:             int  = Field(
        default=5,
        description="Number of chunks to retrieve"
    )
    chunking_strategy: str  = Field(
        default="sliding_window",
        description="Which indexed strategy to query against"
    )
    use_hybrid:        bool = Field(
        default=False,
        description="Use hybrid (dense+sparse) retrieval"
    )
    use_reranker:      bool = Field(
        default=False,
        description="Apply cross-encoder reranking"
    )


class QueryResponse(BaseModel):
    """Response from POST /query."""
    question:          str
    answer:            str
    source_chunks:     list[SourceChunk]
    chunking_strategy: str
    retrieval_method:  str
    latency_ms:        float
    metadata:          dict


# ─────────────────────────────────────────────
# 3. Health endpoint schema
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:     str
    version:    str = "1.0.0"
    components: dict