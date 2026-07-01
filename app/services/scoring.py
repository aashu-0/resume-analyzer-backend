"""
Scoring
=======
"Semantic score (Cosine similarity)" and
"Keyword match (Skill overlap %)"
"""

from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document

from app.schemas import JDKeywords, KeywordMatch, SemanticScore
from app.services.embeddings import get_embeddings


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    # Cosine similarity is in [-1, 1]; clamp and rescale to a 0-100 score.
    similarity = max(-1.0, min(1.0, similarity))
    return round((similarity + 1) / 2 * 100, 2)


def compute_semantic_score(
    resume_documents: list[Document],
    jd_text: str,
) -> SemanticScore:
    """
    Compute overall and per-section cosine similarity between the resume
    content and the job description, using the same Cohere embedding model
    used for indexing (per the "consistent embeddings" rule).
    """
    embeddings = get_embeddings()

    jd_vector = embeddings.embed_query(jd_text)

    # Overall: embed the concatenation of all resume chunks.
    full_resume_text = "\n\n".join(doc.page_content for doc in resume_documents)
    resume_vector = embeddings.embed_query(full_resume_text)
    overall = _cosine_similarity(resume_vector, jd_vector)

    # Per-section: group chunks by section, embed each section's text.
    sections: dict[str, list[str]] = {}
    for doc in resume_documents:
        section = doc.metadata.get("section", "General")
        sections.setdefault(section, []).append(doc.page_content)

    section_similarities: dict[str, float] = {}
    for section, chunks in sections.items():
        section_text = "\n\n".join(chunks)
        section_vector = embeddings.embed_query(section_text)
        section_similarities[section] = _cosine_similarity(section_vector, jd_vector)

    return SemanticScore(overall_similarity=overall, section_similarities=section_similarities)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9.+#\s]", " ", text.lower())


def compute_keyword_match(resume_text: str, jd_keywords: JDKeywords) -> KeywordMatch:
    """
    Compare the JD's required/nice-to-have skills against the resume text.

    overlap_percent is computed against `required_skills` (falling back to
    `keywords` if no required skills were extracted at all).
    """
    normalized_resume = _normalize(resume_text)

    target_skills = jd_keywords.required_skills or jd_keywords.keywords
    if not target_skills:
        return KeywordMatch(overlap_percent=0.0, matched_skills=[], missing_skills=[])

    matched: list[str] = []
    missing: list[str] = []

    for skill in target_skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized_resume):
            matched.append(skill)
        else:
            missing.append(skill)

    overlap_percent = round(len(matched) / len(target_skills) * 100, 2)

    return KeywordMatch(
        overlap_percent=overlap_percent,
        matched_skills=matched,
        missing_skills=missing,
    )
