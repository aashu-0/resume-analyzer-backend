from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze

settings = get_settings()

app = FastAPI(
    title="Resume <-> JD Analyzer",
    description=(
        "RAG-based app that scores a resume against a job description "
        "using semantic similarity, keyword/skill overlap, and an LLM-based "
        "qualitative gap analysis."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "vector_store": settings.vector_store}
