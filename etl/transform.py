"""
ETL Transform Module — Chunking, table classification, metadata tagging.

Text chunks: 600 tokens (tiktoken cl100k_base), 100-token overlap.
Tables: Classified by rule-based type detection, kept as structured records.
"""

import json
import logging
import re
import uuid
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE = 600       # tokens
CHUNK_OVERLAP = 100    # tokens
ENCODING_NAME = "cl100k_base"

# ─── Table Classification Rules ──────────────────────────────────────────────

TABLE_RULES: list[dict[str, Any]] = [
    {
        "type": "growth_projection",
        "keywords": ["growth", "cagr", "2030", "2029", "2028", "projection", "gva employment"],
        "min_matches": 1,
    },
    {
        "type": "regional_distribution",
        "keywords": ["region", "cork", "galway", "dublin", "limerick", "belfast", "county",
                      "offices", "10k population", "10,000 capita"],
        "min_matches": 2,
    },
    {
        "type": "firm_size_count",
        "keywords": ["large", "medium", "small", "micro", "firm", "size", "ftes"],
        "min_matches": 3,
    },
    {
        "type": "firm_classification",
        "keywords": ["dedicated", "diversified", "pure-play", "indigenous", "foreign",
                      "domestic", "foreign-owned"],
        "min_matches": 2,
    },
    {
        "type": "employment_total",
        "keywords": ["employment", "jobs", "headcount", "workforce", "total:", "489",
                      "7,351", "7351"],
        "min_matches": 2,
    },
    {
        "type": "gva_data",
        "keywords": ["gva", "gross value", "€459m", "€617m", "€1.1bn", "average salary",
                      "per employee"],
        "min_matches": 2,
    },
    {
        "type": "benchmark_comparison",
        "keywords": ["benchmark", "comparator", "northern ireland", "uk", "estonia", "israel",
                      "100,000"],
        "min_matches": 2,
    },
    {
        "type": "taxonomy",
        "keywords": ["taxonomy", "managed security", "threat intelligence", "risk compliance",
                      "services offered"],
        "min_matches": 2,
    },
    {
        "type": "percentage_distribution",
        "keywords": ["%", "percent", "share", "proportion"],
        "min_matches": 2,
    },
]


def _classify_table(table: dict[str, Any]) -> str:
    """Classify a table element by inspecting columns, rows, and section text."""
    # Build a searchable text blob from all table contents
    text_blob = " ".join([
        " ".join(table.get("columns", [])),
        table.get("section", ""),
        " ".join(
            " ".join(str(v) for v in row.values())
            for row in table.get("rows", [])[:5]  # First 5 rows only
        ),
    ]).lower()

    best_type = "other"
    best_score = 0

    for rule in TABLE_RULES:
        matches = sum(1 for kw in rule["keywords"] if kw.lower() in text_blob)
        if matches >= rule["min_matches"] and matches > best_score:
            best_score = matches
            best_type = rule["type"]

    return best_type


def _chunk_text(text: str, page: int, section: str, source_file: str) -> list[dict[str, Any]]:
    """Split text into token-bounded chunks with overlap."""
    enc = tiktoken.get_encoding(ENCODING_NAME)
    tokens = enc.encode(text)

    if len(tokens) <= CHUNK_SIZE:
        return [{
            "chunk_id": str(uuid.uuid4()),
            "type": "text",
            "content": text,
            "page": page,
            "section": section,
            "source_file": source_file,
            "table_type": "",
            "token_count": len(tokens),
        }]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "type": "text",
            "content": chunk_text,
            "page": page,
            "section": section,
            "source_file": source_file,
            "table_type": "",
            "token_count": len(chunk_tokens),
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def transform_elements(
    elements: list[dict[str, Any]],
    source_file: str = "report.pdf",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Transform extracted elements into embeddable text chunks and structured table records.

    Returns:
        (text_chunks, table_records)
        - text_chunks: list of dicts ready for vector embedding
        - table_records: list of dicts ready for SQL insertion
    """
    text_chunks: list[dict[str, Any]] = []
    table_records: list[dict[str, Any]] = []

    for elem in elements:
        if elem["type"] == "text":
            chunks = _chunk_text(
                text=elem["content"],
                page=elem["page"],
                section=elem.get("section", ""),
                source_file=source_file,
            )
            text_chunks.extend(chunks)

        elif elem["type"] == "table":
            table_type = _classify_table(elem)
            table_id = str(uuid.uuid4())

            # Table record for SQL storage
            table_records.append({
                "table_id": table_id,
                "page": elem["page"],
                "section": elem.get("section", ""),
                "table_type": table_type,
                "columns": elem.get("columns", []),
                "rows": elem.get("rows", []),
                "raw_json": json.dumps({
                    "columns": elem.get("columns", []),
                    "rows": elem.get("rows", []),
                }),
            })

            # Also create a text chunk for vector embedding (so tables are searchable)
            table_text = _table_to_text(elem)
            if table_text:
                text_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "type": "table",
                    "content": table_text,
                    "page": elem["page"],
                    "section": elem.get("section", ""),
                    "source_file": source_file,
                    "table_type": table_type,
                    "token_count": len(tiktoken.get_encoding(ENCODING_NAME).encode(table_text)),
                })

    logger.info(
        f"Transformed: {len(text_chunks)} text chunks, {len(table_records)} table records"
    )
    return text_chunks, table_records


def _table_to_text(table: dict[str, Any]) -> str:
    """Convert table element to a readable text representation for embedding."""
    parts = []
    if table.get("section"):
        parts.append(f"Section: {table['section']}")
    parts.append(f"Page: {table['page']}")

    columns = table.get("columns", [])
    rows = table.get("rows", [])

    if columns:
        parts.append("Columns: " + " | ".join(columns))

    for row in rows[:20]:  # Cap at 20 rows
        if isinstance(row, dict):
            row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        else:
            row_str = str(row)
        parts.append(row_str)

    return "\n".join(parts)
