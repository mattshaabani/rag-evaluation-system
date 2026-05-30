"""
src/utils/config.py

Central configuration loader for the RAG system.
Loads settings from:
  - .env file (secrets, URLs)
  - configs/rag_config.yaml (ML pipeline settings)
  - configs/eval_config.yaml (evaluation settings)

Every other module imports from here. Nothing else reads .env or yaml directly.
"""

from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
# 1. Project root path
# ─────────────────────────────────────────────

# __file__ is the path to this file: src/utils/config.py
# .parent      → src/utils/
# .parent.parent → src/
# .parent.parent.parent → project root (rag-evaluation-system/)
ROOT_DIR = Path(__file__).parent.parent.parent


# ─────────────────────────────────────────────
# 2. YAML loader helper
# ─────────────────────────────────────────────

def load_yaml(filename: str) -> dict:
    """Load a yaml file from the configs/ directory."""
    path = ROOT_DIR / "configs" / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 3. Load both yaml files once at import time
# ─────────────────────────────────────────────

_rag_cfg  = load_yaml("rag_config.yaml")
_eval_cfg = load_yaml("eval_config.yaml")


# ─────────────────────────────────────────────
# 4. Chunking settings
# ─────────────────────────────────────────────

class ChunkingConfig:
    chunk_size:    int       = _rag_cfg["chunking"]["chunk_size"]
    chunk_overlap: int       = _rag_cfg["chunking"]["chunk_overlap"]
    strategies:    list[str] = _rag_cfg["chunking"]["strategies"]


# ─────────────────────────────────────────────
# 5. Embedding settings
# ─────────────────────────────────────────────

class EmbeddingConfig:
    dense_model:    str   = _rag_cfg["embeddings"]["dense_model"]
    sparse_model:   str   = _rag_cfg["embeddings"]["sparse_model"]
    hybrid_weight:  float = _rag_cfg["embeddings"]["hybrid_weight"]


# ─────────────────────────────────────────────
# 6. Retrieval settings
# ─────────────────────────────────────────────

class RetrievalConfig:
    top_k:          int = _rag_cfg["retrieval"]["top_k"]
    reranker_model: str = _rag_cfg["retrieval"]["reranker_model"]


# ─────────────────────────────────────────────
# 7. Generation settings
# ─────────────────────────────────────────────

class GenerationConfig:
    backend:     str   = _rag_cfg["generation"]["backend"]
    model:       str   = _rag_cfg["generation"]["model"]
    temperature: float = _rag_cfg["generation"]["temperature"]
    max_tokens:  int   = _rag_cfg["generation"]["max_tokens"]


# ─────────────────────────────────────────────
# 8. Vector store settings
# ─────────────────────────────────────────────

class VectorStoreConfig:
    provider:        str = _rag_cfg["vectorstore"]["provider"]
    collection_name: str = _rag_cfg["vectorstore"]["collection_name"]


# ─────────────────────────────────────────────
# 9. Evaluation settings
# ─────────────────────────────────────────────

class EvaluationConfig:
    metrics:                list[str]   = _eval_cfg["evaluation"]["metrics"]
    retrieval_metrics:      list[str]   = _eval_cfg["evaluation"]["retrieval_metrics"]
    k_values:               list[int]   = _eval_cfg["evaluation"]["k_values"]
    faithfulness_threshold: float       = _eval_cfg["evaluation"]["faithfulness_threshold"]
    relevancy_threshold:    float       = _eval_cfg["evaluation"]["relevancy_threshold"]
    eval_dataset_path:      Path        = ROOT_DIR / _eval_cfg["dataset"]["eval_dataset_path"]
    sample_size:            int         = _eval_cfg["dataset"]["sample_size"]


# ─────────────────────────────────────────────
# 10. Environment settings (from .env file)
# ─────────────────────────────────────────────

class EnvSettings(BaseSettings):
    """
    Pydantic reads these from your .env file automatically.
    If a variable is missing from .env, it raises an error at startup.
    """
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"          # ignore unknown vars in .env
    )

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model:    str = Field(default="llama3.1")

    # ChromaDB
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)

    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    # App
    app_env:   str = Field(default="development")
    log_level: str = Field(default="INFO")

    # HuggingFace
    hf_api_token: str = Field(default="")
    hf_api_url:   str = Field(default="https://api-inference.huggingface.co/models")


# ─────────────────────────────────────────────
# 11. Master settings object
# ─────────────────────────────────────────────

class Settings:
    """
    Single object that holds everything.
    Import this anywhere in the project.
    
    Usage:
        from src.utils.config import settings
        print(settings.chunking.chunk_size)   # 512
        print(settings.env.ollama_base_url)   # http://localhost:11434
    """
    chunking:    ChunkingConfig    = ChunkingConfig()
    embedding:   EmbeddingConfig   = EmbeddingConfig()
    retrieval:   RetrievalConfig   = RetrievalConfig()
    generation:  GenerationConfig  = GenerationConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    evaluation:  EvaluationConfig  = EvaluationConfig()
    env:         EnvSettings       = EnvSettings()
    root_dir:    Path              = ROOT_DIR


# This is the one object everything else imports
settings = Settings()