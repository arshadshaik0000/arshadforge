"""
Agent Tools — 4 deterministic tools for the agent graph.

1. vector_retrieval_tool: Semantic search in ChromaDB
2. sql_query_tool: Parameterized SQL against SQLite
3. python_calculator_tool: Deterministic math (CAGR, percentages)
4. citation_validation_tool: Validate numbers appear in source text

All tools are pure Python functions. No LLM calls. All return structured dicts.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))

# Lazy-loaded ChromaDB client
_chroma_collection = None


def _get_collection():
    """Lazy-initialize ChromaDB collection using the shared embedding function.

    IMPORTANT: Must use the exact same embedding function that was used during
    ingestion (etl/load.py).  Both paths import from storage.embeddings.
    """
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        from storage.embeddings import get_embedding_function, COLLECTION_NAME

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        embed_fn = get_embedding_function()

        _chroma_collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    return _chroma_collection


# ─── Tool 1: Vector Retrieval ─────────────────────────────────────────────

def vector_retrieval_tool(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Semantic search against ChromaDB.

    Returns:
        {
            "tool": "vector_retrieval",
            "query": str,
            "results": [{"content", "page", "section", "score", "type"}],
            "count": int
        }
    """
    logger.info(f"[vector_retrieval] query='{query[:80]}...' top_k={top_k}")

    try:
        collection = _get_collection()
        # Always use query_texts — the collection's embedding function handles encoding.
        # This avoids any mismatch between preloaded vs persisted embedders.
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                hits.append({
                    "content": doc,
                    "page": meta.get("page", 0),
                    "section": meta.get("section", ""),
                    "type": meta.get("type", "text"),
                    "table_type": meta.get("table_type", ""),
                    "score": round(1 - distance, 4),  # Convert distance to similarity
                })

        logger.info(f"[vector_retrieval] Found {len(hits)} results")
        return {
            "tool": "vector_retrieval",
            "query": query,
            "results": hits,
            "count": len(hits),
        }

    except Exception as e:
        logger.error(f"[vector_retrieval] Error: {e}")
        return {
            "tool": "vector_retrieval",
            "query": query,
            "results": [],
            "count": 0,
            "error": str(e),
        }


# ─── Tool 2: SQL Query ───────────────────────────────────────────────────

def sql_query_tool(sql: str, params: tuple = ()) -> dict[str, Any]:
    """
    Execute a parameterized SQL query against the SQLite database.

    Security: Only SELECT statements are allowed.

    Returns:
        {
            "tool": "sql_query",
            "sql": str,
            "results": [dict],
            "count": int
        }
    """
    from storage.db import execute_query

    logger.info(f"[sql_query] sql='{sql[:100]}...'")

    try:
        rows = execute_query(sql, params)
        logger.info(f"[sql_query] Returned {len(rows)} rows")
        return {
            "tool": "sql_query",
            "sql": sql,
            "params": list(params),
            "results": rows,
            "count": len(rows),
        }
    except Exception as e:
        logger.error(f"[sql_query] Error: {e}")
        return {
            "tool": "sql_query",
            "sql": sql,
            "params": list(params),
            "results": [],
            "count": 0,
            "error": str(e),
        }


# ─── Tool 3: Python Calculator ──────────────────────────────────────────

def python_calculator_tool(expression: str, variables: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """
    Execute a mathematical expression using Python.

    The LLM MUST NOT perform arithmetic — this tool enforces determinism.

    Supports:
    - Basic math: +, -, *, /, **, ()
    - CAGR: (target/baseline)**(1/years) - 1
    - Percentage: value / total * 100

    Returns:
        {
            "tool": "python_calculator",
            "expression": str,
            "result": float,
            "formatted": str
        }
    """
    logger.info(f"[python_calculator] expression='{expression}' vars={variables}")

    try:
        # Build safe namespace with math functions
        safe_namespace: dict[str, Any] = {
            "abs": abs,
            "round": round,
            "pow": pow,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        if variables:
            safe_namespace.update(variables)

        # Evaluate expression
        result = eval(expression, {"__builtins__": {}}, safe_namespace)

        # Format result
        if isinstance(result, float):
            if abs(result) < 1:  # Likely a percentage/rate
                formatted = f"{result * 100:.2f}%"
            else:
                formatted = f"{result:,.2f}"
        else:
            formatted = str(result)

        logger.info(f"[python_calculator] Result: {result} ({formatted})")
        return {
            "tool": "python_calculator",
            "expression": expression,
            "variables": variables or {},
            "result": float(result),
            "formatted": formatted,
        }

    except Exception as e:
        logger.error(f"[python_calculator] Error: {e}")
        return {
            "tool": "python_calculator",
            "expression": expression,
            "variables": variables or {},
            "result": None,
            "formatted": "ERROR",
            "error": str(e),
        }


# ─── Tool 4: Citation Validation ────────────────────────────────────────

def citation_validation_tool(
    claimed_number: int | float | str,
    source_text: str,
    page: int,
) -> dict[str, Any]:
    """
    Validate that a claimed number appears in the source text.

    Checks:
    - Exact match (with comma/decimal normalization)
    - Tolerance match (within 5% for large numbers)
    - Shorthand expansion (7,350 ≈ 7,351; €1.1bn = 1,100,000,000)

    Returns:
        {
            "tool": "citation_validation",
            "valid": bool,
            "claimed": str,
            "found_numbers": list,
            "page": int,
            "match_type": str  # "exact", "tolerance", "shorthand", "NOT_FOUND"
        }
    """
    logger.info(f"[citation_validation] Validating {claimed_number} on page {page}")

    # First extract all numerical tokens from the claim string so we can
    # ensure each appears in the page before proceeding.
    claim_str = str(claimed_number)
    claim_nums = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?%?", claim_str)
    missing = []
    text_for_search = source_text.replace(",", "")
    for n in claim_nums:
        # ignore pure percentage tokens for now (they are harder to verify)
        if n.endswith("%"):
            continue
        if n.replace(",", "") not in text_for_search:
            missing.append(n)
    if missing:
        # return failure rather than raise to keep agent stable
        return {
            "tool": "citation_validation",
            "valid": False,
            "claimed": claimed_number,
            "found_numbers": [],
            "page": page,
            "match_type": "MISSING_NUMBERS",
            "error": f"Missing numeric tokens: {missing}",
        }

    # Normalize claimed number for downstream matching
    claimed_str = claim_nums[0] if claim_nums else claim_str
    claimed_str = claimed_str.replace(",", "").strip()
    try:
        claimed_float = float(claimed_str)
    except ValueError:
        return {
            "tool": "citation_validation",
            "valid": False,
            "claimed": str(claimed_number),
            "found_numbers": [],
            "page": page,
            "match_type": "INVALID_NUMBER",
            "error": f"Cannot parse '{claimed_number}' as number",
        }

    # Extract all numbers from source text
    raw_numbers = re.findall(r"[\d,]+\.?\d*", source_text)
    found_numbers = []
    for n in raw_numbers:
        try:
            found_numbers.append(float(n.replace(",", "")))
        except ValueError:
            continue

    # Also expand shorthand numbers (e.g., "7,350" → 7350, "€1.1bn" → 1100000000)
    shorthand_pattern = r"([\d,]+\.?\d*)\s*(bn|billion|million|mn|m|k|thousand)\b"
    for match in re.finditer(shorthand_pattern, source_text, re.IGNORECASE):
        base = float(match.group(1).replace(",", ""))
        suffix = match.group(2).lower()
        multipliers = {"bn": 1e9, "billion": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6,
                        "k": 1e3, "thousand": 1e3}
        if suffix in multipliers:
            found_numbers.append(base * multipliers[suffix])

    # Check for exact match
    for fn in found_numbers:
        if abs(fn - claimed_float) < 0.01:
            return {
                "tool": "citation_validation",
                "valid": True,
                "claimed": str(claimed_number),
                "found_numbers": [f"{n:,.0f}" for n in found_numbers[:10]],
                "page": page,
                "match_type": "exact",
            }

    # Check for tolerance match (within 5% for numbers > 100)
    if claimed_float > 100:
        for fn in found_numbers:
            if fn > 0 and abs(fn - claimed_float) / claimed_float < 0.05:
                return {
                    "tool": "citation_validation",
                    "valid": True,
                    "claimed": str(claimed_number),
                    "found_numbers": [f"{n:,.0f}" for n in found_numbers[:10]],
                    "page": page,
                    "match_type": "tolerance",
                    "closest_match": f"{fn:,.0f}",
                }

    # Not found
    return {
        "tool": "citation_validation",
        "valid": False,
        "claimed": str(claimed_number),
        "found_numbers": [f"{n:,.0f}" for n in found_numbers[:10]],
        "page": page,
        "match_type": "NOT_FOUND",
    }
