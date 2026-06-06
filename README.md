# RAG Evaluation System

A production-ready Retrieval Augmented Generation (RAG) pipeline with a comprehensive evaluation framework for comparing chunking strategies.

---

## Overview

This project implements a complete RAG system from scratch, with the core focus on **scientific evaluation** — measuring which chunking strategy produces the best retrieval and answer quality using real metrics.

**What makes this different from a typical RAG tutorial:**
- Three chunking strategies implemented and compared scientifically
- Full IR evaluation metrics (NDCG, MRR, Precision@K, Recall@K)
- RAGAS-style answer quality metrics (faithfulness, relevancy)
- MLflow experiment tracking with visual comparison dashboard
- Production-ready API with monitoring

---

## Architecture

    Documents (PDF, TXT, URL)
            |
       DocumentLoader
            |
       Chunker (3 strategies)
            |
       DenseEmbedder + BM25Retriever
            |
       ChromaDB / Qdrant
            |
       HybridRetriever + RRF Fusion
            |
       CrossEncoder Reranker
            |
       RAGPromptTemplate
            |
       LLM (Anthropic Claude / Ollama)
            |
       Answer + Sources
            |
       RAGAS Evaluation + MLflow Tracking

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Sparse retrieval | BM25 custom implementation |
| Vector stores | ChromaDB, Qdrant |
| LLM backend | Anthropic Claude, Ollama |
| Evaluation | RAGAS-style + IR metrics |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Environment | Conda + ipykernel |

---

## Project Structure

    rag-evaluation-system/
    |-- src/
    |   |-- data/
    |   |   |-- loader.py            # PDF, TXT, URL loaders
    |   |   |-- chunker.py           # Sliding window, semantic, hierarchical
    |   |   └-- preprocessor.py
    |   |-- embeddings/
    |   |   |-- dense.py             # Sentence transformers
    |   |   |-- sparse.py            # BM25 implementation
    |   |   └-- hybrid.py            # RRF fusion
    |   |-- vectorstore/
    |   |   |-- base.py              # Abstract interface
    |   |   |-- chroma_store.py      # ChromaDB implementation
    |   |   └-- qdrant_store.py      # Qdrant implementation
    |   |-- retrieval/
    |   |   |-- dense_retriever.py   # Dense RAG pipeline
    |   |   |-- hybrid_retriever.py  # Hybrid RAG pipeline
    |   |   └-- reranker.py          # Cross-encoder reranker
    |   |-- generation/
    |   |   |-- prompt_templates.py  # RAG, eval, query rewrite prompts
    |   |   |-- ollama_client.py     # Ollama + factory function
    |   |   |-- hf_client.py         # HuggingFace Inference API
    |   |   └-- anthropic_client.py  # Anthropic Claude API
    |   |-- evaluation/
    |   |   |-- retrieval_metrics.py # NDCG, MRR, Precision, Recall
    |   |   |-- ragas_eval.py        # Faithfulness, relevancy
    |   |   |-- ab_test.py           # MLflow A/B test runner
    |   |   └-- report.py            # Comparison report generator
    |   └-- api/
    |       |-- main.py              # App entry point + middleware
    |       |-- routes.py            # Endpoint handlers
    |       └-- schemas.py           # Pydantic request/response models
    |-- notebooks/
    |   └-- run_ab_test.py           # A/B test runner script
    |-- configs/
    |   |-- rag_config.yaml          # Pipeline settings
    |   └-- eval_config.yaml         # Evaluation settings
    |-- data/
    |   |-- raw/                     # Input documents
    |   |-- processed/               # Chunked output
    |   └-- eval_datasets/           # Q&A pairs and reports
    |-- docker/
    |   └-- Dockerfile               # Production image
    |-- monitoring/
    |   └-- prometheus.yml           # Prometheus scrape config
    |-- tests/                       # pytest test suite
    |-- docker-compose.yml           # Full stack deployment
    |-- Makefile                     # Common commands
    |-- environment.yml              # Conda environment
    |-- requirements.txt             # Pip dependencies
    └-- .env.example                 # Environment variable template

---

## Quick Start

**1. Clone and set up environment**

    git clone https://github.com/yourusername/rag-evaluation-system.git
    cd rag-evaluation-system

    conda env create -f environment.yml
    conda activate rag-eval
    pip install -e .
    python -m ipykernel install --user --name rag-eval --display-name "Python (rag-eval)"

**2. Configure environment variables**

    cp .env.example .env

Edit .env and add your API keys.

**3. Run the API locally**

    uvicorn src.api.main:app --reload --port 8080

Open http://localhost:8080/docs for the interactive Swagger UI.

**4. Run the A/B evaluation**

    python notebooks/run_ab_test.py
    mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db

Open http://localhost:5000 to see the MLflow comparison dashboard.

**5. Run with Docker (full stack)**

    docker-compose up --build

---

## Evaluation Results

A/B test comparing three chunking strategies on 10 evaluation questions:

| Metric | Sliding Window | Semantic | Hierarchical | Winner |
|---|---|---|---|---|
| NDCG@5 | 0.9262 | 0.2161 | 1.2709 | Hierarchical |
| MRR | 0.9000 | 0.1600 | 0.6500 | Sliding Window |
| Precision@5 | 0.2000 | 0.0800 | 0.4000 | Hierarchical |
| Recall@5 | 1.0000 | 0.4000 | 2.0000 | Hierarchical |
| Faithfulness | 0.4190 | 0.6305 | 0.5140 | Semantic |
| Answer Relevancy | 0.4839 | 0.4839 | 0.4839 | Tied |
| Context Precision | 0.2058 | 0.3915 | 0.2198 | Semantic |

**Key finding:** Hierarchical chunking wins on retrieval volume metrics while Semantic chunking wins on precision and faithfulness. The right choice depends on your use case — recall-heavy tasks favor hierarchical, precision-heavy tasks favor semantic.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/v1/health | System health check |
| POST | /api/v1/ingest | Load and index documents |
| POST | /api/v1/query | Ask a question, get an answer |
| GET | /metrics | Prometheus metrics |
| GET | /docs | Swagger UI |

---

## Math Behind the System

**Vector Embeddings — Cosine Similarity**

Text is converted to 384-dimensional vectors. Similar texts produce similar vectors:

    cosine_similarity(A, B) = (A · B) / (||A|| x ||B||)

**BM25 Sparse Retrieval**

Keyword-based scoring using term frequency and inverse document frequency:

    BM25(t,d) = IDF(t) x tf(t,d) x (k1+1) / (tf(t,d) + k1 x (1 - b + b x |d|/avgdl))

**Reciprocal Rank Fusion**

Combining dense and sparse rankings without score normalization:

    RRF_score(doc) = sum( 1 / (k + rank_i(doc)) )

**NDCG — Position-weighted retrieval quality**

    DCG@K  = sum( rel_i / log2(i+1) )
    NDCG@K = DCG@K / IDCG@K

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| ANTHROPIC_API_KEY | Anthropic Claude API key | required |
| OLLAMA_BASE_URL | Ollama server URL | http://localhost:11434 |
| CHROMA_HOST | ChromaDB host | localhost |
| QDRANT_HOST | Qdrant host | localhost |
| LOG_LEVEL | Logging level | INFO |

---

## License

MIT