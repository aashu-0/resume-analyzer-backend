"""
Vector Store
============
Each analyzed resume gets its own isolated namespace (Pinecone) keyed by `resume_id`, so multiple
concurrent analyses never mix embeddings -- mirroring the
"session-based namespace" pattern.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import get_settings
from app.services.embeddings import get_embeddings


def _build_pinecone(resume_id: str, documents: list[Document]) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )

    index = pc.Index(settings.pinecone_index_name)
    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=resume_id,
    )
    vectorstore.add_documents(documents)
    return vectorstore


def build_vectorstore(resume_id: str, documents: list[Document]) -> VectorStore:
    """
    Embed `documents` and index them into the configured vector store
    (Pinecone), isolated by
    `resume_id`.
    """
    if not documents:
        raise ValueError("No documents to index -- resume parsing/chunking produced nothing.")
    
    return _build_pinecone(resume_id, documents)


def cleanup_vectorstore(resume_id: str) -> None:
    """
    Best-effort cleanup of resources created for a resume_id.

    - Pinecone: deletes all vectors in the resume's namespace.
    """
    settings = get_settings()

    if settings.vector_store == "pinecone":
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index_name)
            index.delete(delete_all=True, namespace=resume_id)
        except Exception:
            pass
        return
