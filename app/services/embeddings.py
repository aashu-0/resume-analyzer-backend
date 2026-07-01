"""
Embedding Model
===============
the embedding model instance/config must be used for both indexing and querying.
This module provides a single cached factory function so every other module
(vector store, retriever, scoring) shares the exact same embeddings config.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_cohere import CohereEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> CohereEmbeddings:
    """Return a cached CohereEmbeddings instance (e.g. embed-english-v3.0)."""
    settings = get_settings()
    return CohereEmbeddings(
        model=settings.cohere_embedding_model,
        cohere_api_key=settings.cohere_api_key,
    )
