"""
LLM Judge
=========
Takes the chunks returned by the Retriever (top-k semantic search over the
resume) along with the JD text and produces a structured qualitative
assessment: fit summary, strengths, gaps, and rewrite suggestions.

Uses LangChain's `with_structured_output` so the response is parsed
directly into the `GapAnalysis` pydantic model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.schemas import GapAnalysis, JDKeywords

_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter and resume coach. Given "
            "excerpts from a candidate's resume and a job description, "
            "evaluate how well the candidate fits the role. Be specific and "
            "actionable. Do not invent experience that isn't in the resume "
            "excerpts.",
        ),
        (
            "human",
            "JOB DESCRIPTION:\n{jd_text}\n\n"
            "REQUIRED SKILLS: {required_skills}\n\n"
            "RELEVANT RESUME EXCERPTS (retrieved via top-k semantic search):\n"
            "{resume_excerpts}\n\n"
            "Provide:\n"
            "1. fit_summary: a 2-3 sentence overall assessment of fit for this role.\n"
            "2. strengths: bullet list of the candidate's strongest matches to the JD.\n"
            "3. gaps: bullet list of missing or weak areas relative to the JD.\n"
            "4. rewrite_suggestions: concrete suggestions for rewriting/adding "
            "resume content to better target this JD.",
        ),
    ]
)


def _format_excerpts(documents: list[Document]) -> str:
    parts = []
    for doc in documents:
        section = doc.metadata.get("section", "General")
        parts.append(f"[{section}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _fallback_gap_analysis(missing_skills: list[str]) -> GapAnalysis:
    """Used when no GOOGLE_API_KEY is configured or the LLM call fails."""
    return GapAnalysis(
        fit_summary=(
            "LLM-based qualitative analysis is unavailable (no GOOGLE_API_KEY "
            "configured or the Gemini call failed). Falling back to "
            "keyword/semantic scores only."
        ),
        strengths=[],
        gaps=[f"Missing or unverified skill: {skill}" for skill in missing_skills],
        rewrite_suggestions=[
            "Configure GOOGLE_API_KEY to enable AI-generated rewrite suggestions."
        ],
    )


def run_llm_judge(
    retrieved_chunks: list[Document],
    jd_text: str,
    jd_keywords: JDKeywords,
) -> GapAnalysis:
    """Run the qualitative gap-analysis LLM judge over retrieved resume excerpts."""
    settings = get_settings()

    if not settings.google_api_key:
        return _fallback_gap_analysis([])

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
        structured_llm = llm.with_structured_output(GapAnalysis)

        chain = _JUDGE_PROMPT | structured_llm
        result = chain.invoke(
            {
                "jd_text": jd_text[:6000],
                "required_skills": ", ".join(jd_keywords.required_skills) or "Not specified",
                "resume_excerpts": _format_excerpts(retrieved_chunks)[:6000],
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 - never let the judge break the whole pipeline
        fallback = _fallback_gap_analysis([])
        fallback.gaps.append(f"LLM judge error: {exc}")
        return fallback
