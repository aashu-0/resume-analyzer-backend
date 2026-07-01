"""
JD Preprocessor
================
Responsibilities:
  1. Load the job description, either from raw pasted text or a URL
     (mirrors the "Loading Web Pages" pattern from the langchain-rag skill,
     implemented with requests + BeautifulSoup to avoid extra heavy deps).
  2. Extract a structured set of skills/keywords using:
       a) a curated technical-skills dictionary (fast, deterministic), and
       b) an LLM-based extraction pass (Gemini) for required vs. nice-to-have
          skills and free-form keywords, with a graceful fallback if the
          LLM call fails or no API key is configured.
"""

from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.schemas import JDKeywords

# A reasonably broad curated list of technical / role skills commonly seen in
# JDs. This acts as the deterministic fallback / supplement to LLM extraction
# and is also used for the Keyword Match scorer.
SKILL_DICTIONARY: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    # Web / backend
    "fastapi", "django", "flask", "node.js", "express", "rest api", "graphql",
    "microservices", "websockets",
    # Frontend
    "react", "vue", "angular", "next.js", "tailwind",
    # Data / ML
    "machine learning", "deep learning", "nlp", "llm", "rag",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "langchain", "langgraph", "transformers", "huggingface",
    # MLOps / infra
    "docker", "kubernetes", "ci/cd", "aws", "gcp", "azure",
    "terraform", "mlflow", "dvc", "prometheus", "grafana",
    # Databases / vector stores
    "postgresql", "mysql", "mongodb", "redis", "pinecone", "faiss", "chroma",
    # Other
    "git", "linux", "agile", "system design", "data structures", "algorithms",
]


def _dictionary_keywords(jd_text: str) -> list[str]:
    """Match the JD text against the curated skill dictionary (case-insensitive)."""
    jd_lower = jd_text.lower()
    found = []
    for skill in SKILL_DICTIONARY:
        # Word-boundary-ish match so "go" doesn't match inside "ongoing", etc.
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, jd_lower):
            found.append(skill)
    return found


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured skill information from job descriptions. "
            "Respond with ONLY valid JSON, no markdown fences, matching this schema:\n"
            '{{"required_skills": [string], "nice_to_have_skills": [string], '
            '"keywords": [string]}}\n'
            "required_skills = must-have technical skills/tools explicitly required. "
            "nice_to_have_skills = skills mentioned as preferred/bonus. "
            "keywords = other important terms (domains, methodologies, soft skills).",
        ),
        ("human", "Job description:\n\n{jd_text}"),
    ]
)


def _llm_extract_keywords(jd_text: str) -> JDKeywords | None:
    """Use Gemini to extract required/nice-to-have skills and keywords. Returns None on failure."""
    settings = get_settings()
    if not settings.google_api_key:
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        chain = _EXTRACTION_PROMPT | llm
        response = chain.invoke({"jd_text": jd_text[:8000]})

        content = response.content.strip()
        # Defensively strip markdown code fences if the model adds them anyway.
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)

        data = json.loads(content)
        return JDKeywords(
            required_skills=[s.lower() for s in data.get("required_skills", [])],
            nice_to_have_skills=[s.lower() for s in data.get("nice_to_have_skills", [])],
            keywords=[s.lower() for s in data.get("keywords", [])],
            raw_text=jd_text,
        )
    except Exception:
        # LLM extraction is a "nice to have" enhancement; never fail the
        # whole pipeline because of it.
        return None


def extract_jd_keywords(jd_text: str) -> JDKeywords:
    """
    Extract structured skills/keywords from a job description.

    Combines a deterministic dictionary pass with an optional LLM pass.
    The dictionary results are merged into `required_skills` so that
    Keyword Match scoring works even without an LLM key configured.
    """
    dictionary_hits = _dictionary_keywords(jd_text)

    llm_result = _llm_extract_keywords(jd_text)
    if llm_result is not None:
        merged_required = sorted(set(llm_result.required_skills) | set(dictionary_hits))
        return JDKeywords(
            required_skills=merged_required,
            nice_to_have_skills=llm_result.nice_to_have_skills,
            keywords=llm_result.keywords,
            raw_text=jd_text,
        )

    # Fallback: dictionary-only extraction.
    return JDKeywords(
        required_skills=dictionary_hits,
        nice_to_have_skills=[],
        keywords=[],
        raw_text=jd_text,
    )
