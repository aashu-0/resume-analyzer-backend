"""
Retriever
=========
Feeds documents into
all three scorers downstream: Semantic Score, Keyword Match, and LLM Judge.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import get_settings


def get_retriever(vectorstore: VectorStore, k: int | None = None):
    """Return a LangChain retriever configured for top-k similarity search."""
    settings = get_settings()
    top_k = k or settings.retriever_top_k
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve_relevant_chunks(vectorstore: VectorStore, query: str, k: int | None = None) -> list[Document]:
    """Run top-k semantic search against the resume vector store for a JD-derived query."""
    retriever = get_retriever(vectorstore, k=k)
    return retriever.invoke(query)
