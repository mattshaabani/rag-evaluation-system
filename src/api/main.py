"""
src/api/main.py

FastAPI application entry point.
Wires together the router, middleware, and startup events.

Run locally:
    uvicorn src.api.main:app --reload --port 8080

API docs available at:
    http://localhost:8080/docs      ← Swagger UI
    http://localhost:8080/redoc     ← ReDoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.routes import router
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# 1. Create FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Evaluation System",
    description="Production-ready RAG pipeline with evaluation framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# 2. Middleware
# ─────────────────────────────────────────────

# CORS — allows browser frontends to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 3. Prometheus metrics
# ─────────────────────────────────────────────

# This automatically tracks:
#   - Request count per endpoint
#   - Request latency (p50, p95, p99)
#   - Error rates
# Exposed at GET /metrics for Prometheus to scrape
Instrumentator().instrument(app).expose(app)

# ─────────────────────────────────────────────
# 4. Include routes
# ─────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")

# ─────────────────────────────────────────────
# 5. Startup event
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(f"RAG API starting up", extra={
        "environment": settings.env.app_env,
        "log_level":   settings.env.log_level,
    })

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG API shutting down")