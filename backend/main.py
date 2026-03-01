"""
FastAPI Backend — Production-grade API for the Agentic Intelligence System.

Endpoints:
    POST /ingest   — Run ETL pipeline on uploaded PDF
    POST /query    — Execute agent graph against query
    GET  /health   — System health check

Architecture:
    - Agent graph loaded at startup
    - ChromaDB + SQLite initialized during ETL
    - Ollama LLM used ONLY for final answer composition
    - Full execution traces saved to logs/
"""

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# embedder will be set during startup to avoid cold-load delays
embedder = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

LOG_DIR = os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Pydantic Models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query string")


class SourceInfo(BaseModel):
    page: int
    section: str = ""
    quote: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    tools_used: list[str] = []
    trace_id: str = ""
    validation_errors: list[str] = []
    confidence: float = 1.0


class IngestResponse(BaseModel):
    status: str
    filename: str
    elements_extracted: int = 0
    text_chunks_created: int = 0
    table_records_created: int = 0
    vectors_stored: int = 0
    tables_stored: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    chromadb_ready: bool = False
    sqlite_ready: bool = False
    ollama_ready: bool = False
    document_ingested: bool = False
    # indicates whether the backend is running in strict citation mode
    strict_sql_sources: bool = False


# ─── App Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("ArshadForge Agentic Intelligence Backend — Starting")
    logger.info("=" * 60)

    # preload embedding model via shared module (singleton, loaded once)
    try:
        from storage.embeddings import warmup
        warmup()
        logger.info("Embedding model preloaded via shared module")
    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}")

    # Ensure directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PROJECT_ROOT / "data", exist_ok=True)

    # Initialize agent graph
    from agents.graph import AgentGraph
    app.state.agent = AgentGraph()
    logger.info("Agent graph initialized")

    # Check if database is already seeded
    try:
        from storage.db import initialize_db
        initialize_db()
        logger.info("SQLite database ready")
    except Exception as e:
        logger.warning(f"SQLite initialization deferred: {e}")

    # echo configuration flags
    strict_mode = os.getenv("STRICT_SQL_SOURCES", "false").lower() == "true"
    logger.info(f"Strict SQL source mode: {strict_mode}")
    app.state.strict_sql_sources = strict_mode

    yield

    logger.info("Shutting down...")


# ─── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="ArshadForge — Agentic Intelligence Backend",
    description=(
        "Production-grade autonomous intelligence system for the Cyber Ireland 2022 Report. "
        "Uses deterministic tool pipelines, hybrid storage (ChromaDB + SQLite), "
        "and LLM-isolated answer composition."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    force_recreate: bool = Query(False, description="Force recreate vector store"),
):
    """
    Run the ETL pipeline on an uploaded PDF.

    Steps:
    1. Save uploaded file
    2. Extract text + tables (pdfplumber)
    3. Transform: chunk text + classify tables
    4. Load: ChromaDB (vectors) + SQLite (structured data)
    """
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    logger.info(f"Received file: {file.filename} ({len(content)} bytes)")

    try:
        from etl.load import run_etl
        summary = run_etl(file_path)

        return IngestResponse(
            status="success",
            filename=file.filename,
            elements_extracted=summary.get("elements_extracted", 0),
            text_chunks_created=summary.get("text_chunks_created", 0),
            table_records_created=summary.get("table_records_created", 0),
            vectors_stored=summary.get("vectors_stored", 0),
            tables_stored=summary.get("tables_stored", 0),
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Execute the agent graph for a given query.

    The agent:
    1. Classifies intent (verification / data_synthesis / forecasting / general)
    2. Executes deterministic tool pipeline
    3. Composes answer using LLM (formatting only)
    4. Validates citations
    5. Returns structured response + trace ID
    """
    agent: Any = app.state.agent

    try:
        result = agent.run(request.query)

        # Save named trace for known test queries
        _save_named_trace(request.query, result.trace)

        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            tools_used=result.tools_used,
            trace_id=result.trace_id,
            validation_errors=result.validation_errors,
            confidence=getattr(result, "confidence", 1.0),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — ChromaDB, SQLite, Ollama connectivity."""
    status_checks = {
        "chromadb_ready": False,
        "sqlite_ready": False,
        "ollama_ready": False,
        "document_ingested": False,
    }

    # Check ChromaDB
    try:
        import chromadb
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection("cyber_ireland_2022")
        count = collection.count()
        status_checks["chromadb_ready"] = True
        status_checks["document_ingested"] = count > 0
    except Exception:
        pass

    # Check SQLite
    try:
        from storage.db import get_all_tables
        tables = get_all_tables()
        status_checks["sqlite_ready"] = len(tables) > 0
    except Exception:
        pass

    # Check Ollama
    try:
        import httpx
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        status_checks["ollama_ready"] = resp.status_code == 200
    except Exception:
        pass

    overall = "ok" if all(status_checks.values()) else "degraded"

    return HealthResponse(
        status=overall,
        strict_sql_sources=app.state.strict_sql_sources,
        **status_checks, # pyright: ignore[reportArgumentType]
    )


# ─── Helper Functions ─────────────────────────────────────────────────────

def _save_named_trace(query: str, trace: dict[str, Any]) -> None:
    """Save a named trace file for known test queries."""
    query_lower = query.lower()

    name_map = {
        "total number of jobs": "test1_trace",
        "pure-play": "test2_trace",
        "cagr": "test3_trace",
        "compound annual growth": "test3_trace",
    }

    for keyword, name in name_map.items():
        if keyword in query_lower:
            trace_path = Path(LOG_DIR) / f"{name}.json"
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2, default=str)
            logger.info(f"Named trace saved: {trace_path}")
            return
