from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class ResumeSection(BaseModel):
    """A single labeled section extracted from a resume (Skills, Experience, etc.)."""

    section: str
    content: str


class JDKeywords(BaseModel):
    """Skills / keywords extracted from a job description."""

    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class SemanticScore(BaseModel):
    overall_similarity: float = Field(..., ge=0, le=100)
    section_similarities: dict[str, float] = Field(default_factory=dict)


class KeywordMatch(BaseModel):
    overlap_percent: float = Field(..., ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class GapAnalysis(BaseModel):
    """Structured output produced by the LLM judge."""

    fit_summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final report / response
# ---------------------------------------------------------------------------

class ScoreReport(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    semantic_score: SemanticScore
    keyword_match: KeywordMatch
    gap_analysis: GapAnalysis
    section_scores: dict[str, float] = Field(default_factory=dict)
    missing_skills: list[str] = Field(default_factory=list)
    rewrite_tips: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    resume_id: str
    sections_detected: list[str]
    jd_keywords: JDKeywords
    report: ScoreReport


class JDInput(BaseModel):
    """Optional JSON body alternative to the multipart form fields."""

    jd_text: Optional[str] = None
