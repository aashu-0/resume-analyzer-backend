"""
Pipeline
========
Orchestrates the full architecture end-to-end:

  INPUT      -> Resume upload + Job description
  INGESTION  -> Document parser -> Section chunker
              -> JD preprocessor (extract skills & keywords)
  RAG CORE   -> Embedding model -> Vector store -> Retriever (top-k)
  SCORING    -> Semantic score (cosine similarity)
              -> Keyword match (skill overlap %)
              -> LLM judge (qualitative gap analysis)
  OUTPUT     -> Score report + recommendations
"""

from __future__ import annotations

import uuid

from app.schemas import AnalyzeResponse, ScoreReport
from app.services import jd_preprocessor
from app.services.chunker import chunk_resume_text
from app.services.document_parser import parse_resume
from app.services.llm_judge import run_llm_judge
from app.services.retriever import retrieve_relevant_chunks
from app.services.scoring import compute_keyword_match, compute_semantic_score
from app.services.vectorstore import build_vectorstore, cleanup_vectorstore

# Weighting for the overall blended score.
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5


def _build_retrieval_query(jd_keywords) -> str:
    """Build a focused query for top-k retrieval from the extracted JD skills."""
    skills = jd_keywords.required_skills + jd_keywords.nice_to_have_skills + jd_keywords.keywords
    if skills:
        return "Relevant experience with: " + ", ".join(skills)
    # Fall back to the raw JD text (truncated) if no keywords were extracted.
    return jd_keywords.raw_text[:1000]


def analyze_resume(
    resume_bytes: bytes,
    filename: str,
    jd_text: str,
    cleanup_after: bool = True,
) -> AnalyzeResponse:
    """
    Run the full Resume <-> JD analysis pipeline and return a structured
    score report + recommendations.
    """
    resume_id = uuid.uuid4().hex

    # --- INGESTION -------------------------------------------------------
    resume_text = parse_resume(resume_bytes, filename)
    sections, documents = chunk_resume_text(resume_text)
    jd_keywords = jd_preprocessor.extract_jd_keywords(jd_text)

    # --- RAG CORE ----------------------------------------------------------
    vectorstore = build_vectorstore(resume_id, documents)
    retrieval_query = _build_retrieval_query(jd_keywords)
    retrieved_chunks = retrieve_relevant_chunks(vectorstore, retrieval_query)

    # --- SCORING -------------------------------------------------------
    semantic_score = compute_semantic_score(documents, jd_text)
    keyword_match = compute_keyword_match(resume_text, jd_keywords)
    gap_analysis = run_llm_judge(retrieved_chunks, jd_text, jd_keywords)

    # --- OUTPUT: Score report + recommendations -------------------------
    overall_score = round(
        semantic_score.overall_similarity * SEMANTIC_WEIGHT
        + keyword_match.overlap_percent * KEYWORD_WEIGHT,
        2,
    )

    # Section scores double as a quick "where to focus" map for the UI.
    section_scores = dict(semantic_score.section_similarities)

    missing_skills = sorted(set(keyword_match.missing_skills))

    rewrite_tips = list(gap_analysis.rewrite_suggestions)
    if missing_skills:
        rewrite_tips.append(
            "Consider adding evidence of these skills if you have relevant "
            "experience: " + ", ".join(missing_skills[:10])
        )

    report = ScoreReport(
        overall_score=overall_score,
        semantic_score=semantic_score,
        keyword_match=keyword_match,
        gap_analysis=gap_analysis,
        section_scores=section_scores,
        missing_skills=missing_skills,
        rewrite_tips=rewrite_tips,
    )

    if cleanup_after:
        cleanup_vectorstore(resume_id)

    return AnalyzeResponse(
        resume_id=resume_id,
        sections_detected=[s.section for s in sections],
        jd_keywords=jd_keywords,
        report=report,
    )
