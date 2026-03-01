"""
Shared Embedding Module — Single source of truth for ChromaDB embedding function.

IMPORTANT: Both ingestion (etl/load.py) and retrieval (agents/tools.py) MUST
use the SAME embedding function instance from this module.  Using different
embedding wrappers causes ChromaDB to reject queries with:
    "Embedding function conflict: new: X vs persisted: Y"
"""

from chromadb.utils import embedding_functions

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cyber_ireland_2022"

# Module-level singleton — created once, reused everywhere.
_embed_fn = None


def get_embedding_function():
    """Return the shared ChromaDB-native SentenceTransformerEmbeddingFunction.

    This is a singleton: the underlying model is loaded once and reused.
    Both ingestion and query paths MUST call this function rather than
    creating their own embedding objects.
    """
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME,
        )
    return _embed_fn


def warmup():
    """Pre-load the embedding model.  Call during server startup."""
    get_embedding_function()
