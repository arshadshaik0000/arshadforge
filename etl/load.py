"""
ETL Load Module — Orchestrates full ETL pipeline.

1. Extracts elements from PDF via extract.py
2. Transforms into text chunks + table records via transform.py
3. Loads text chunks into ChromaDB (vector embeddings)
4. Loads table records into SQLite (structured storage)

Usage:
    python -m etl.load
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.extract import extract_pdf
from etl.transform import transform_elements
from storage.db import initialize_db, insert_table_data, get_db_path

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))


def _load_vectors(text_chunks: list[dict], persist_dir: str) -> int:
    """Load text chunks into ChromaDB with sentence-transformer embeddings."""
    import chromadb
    from storage.embeddings import get_embedding_function, COLLECTION_NAME

    logger.info(f"Initializing ChromaDB at {persist_dir}")

    client = chromadb.PersistentClient(path=persist_dir)

    # Use the shared embedding function — same one used by agents/tools.py
    embed_fn = get_embedding_function()

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch upsert chunks
    batch_size = 50
    total = len(text_chunks)

    for i in range(0, total, batch_size):
        batch = text_chunks[i : i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = [
            {
                "page": c["page"],
                "type": c["type"],
                "section": c["section"],
                "source_file": c["source_file"],
                "table_type": c.get("table_type", ""),
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"  Loaded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

    count = collection.count()
    logger.info(f"ChromaDB collection 'cyber_ireland_2022' now has {count} documents")
    return count


def _load_tables(table_records: list[dict]) -> int:
    """Load structured table records into SQLite."""
    initialize_db()
    count = 0
    for record in table_records:
        try:
            insert_table_data(record)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to insert table record (page {record.get('page')}): {e}")
    logger.info(f"Loaded {count}/{len(table_records)} table records into SQLite")
    return count


from typing import Union

def run_etl(pdf_path: Union[str, Path, None] = None) -> dict:
    """
    Run the full ETL pipeline.

    Args:
        pdf_path: Path to PDF file. If None, looks for the default report in data/raw/

    Returns:
        Summary dict with counts of extracted, transformed, and loaded items.
    """
    if pdf_path is None:
        # Try default locations
        candidates = [
            PROJECT_ROOT / "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf",
            PROJECT_ROOT / "data" / "raw" / "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf",
        ]
        for c in candidates:
            if c.exists():
                pdf_path = c
                break
        if pdf_path is None:
            raise FileNotFoundError(
                "PDF not found. Place it at project root or data/raw/ directory."
            )

    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    logger.info(f"{'='*60}")
    logger.info(f"ETL Pipeline Starting: {source_file}")
    logger.info(f"{'='*60}")

    # Step 1: Extract
    logger.info("\n[1/3] EXTRACT — Parsing PDF...")
    elements = extract_pdf(pdf_path)

    # Step 2: Transform
    logger.info("\n[2/3] TRANSFORM — Chunking text, classifying tables...")
    text_chunks, table_records = transform_elements(elements, source_file)

    # Step 3: Load
    logger.info("\n[3/3] LOAD — Storing in ChromaDB + SQLite...")

    # Ensure data directories exist
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(Path(get_db_path()).parent, exist_ok=True)

    vector_count = _load_vectors(text_chunks, CHROMA_PERSIST_DIR)
    table_count = _load_tables(table_records)

    summary = {
        "pdf_file": str(pdf_path),
        "elements_extracted": len(elements),
        "text_chunks_created": len(text_chunks),
        "table_records_created": len(table_records),
        "vectors_stored": vector_count,
        "tables_stored": table_count,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"ETL Pipeline Complete")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"{'='*60}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_etl()
