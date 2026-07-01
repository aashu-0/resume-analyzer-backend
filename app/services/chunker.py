"""
Section Chunker
===============
=

Step 1 (section detection): split the raw resume text on common resume
header lines (SKILLS, EXPERIENCE, EDUCATION, PROJECTS, ...).

Step 2 (chunking): run each section's content through LangChain's
RecursiveCharacterTextSplitter (chunk_size 500-1500, chunk_overlap ~10-20%) so the resulting Documents are
small enough for embedding while preserving section metadata for retrieval
and per-section scoring.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.schemas import ResumeSection

# Common resume section headers. Order doesn't matter for matching, but the
# canonical names are used for normalized section labels.
SECTION_HEADERS: dict[str, list[str]] = {
    "Summary": ["summary", "objective", "profile", "about me"],
    "Skills": ["skills", "technical skills", "core competencies", "technologies"],
    "Experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "internships",
    ],
    "Projects": ["projects", "academic projects", "personal projects"],
    "Education": ["education", "academic background", "qualifications"],
    "Certifications": ["certifications", "certificates", "licenses"],
    "Achievements": ["achievements", "awards", "honors", "accomplishments"],
    "Publications": ["publications", "research"],
    "Extracurricular": ["extracurricular", "activities", "leadership"],
}

# Build a single regex that matches any of the known header phrases on a
# line by itself (optionally followed by ':' and/or in caps/title case).
_HEADER_ALTERNATION = "|".join(
    re.escape(alias) for aliases in SECTION_HEADERS.values() for alias in aliases
)
_HEADER_RE = re.compile(
    rf"^\s*({_HEADER_ALTERNATION})\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _canonical_section(header_text: str) -> str:
    header_lower = header_text.strip().lower()
    for canonical, aliases in SECTION_HEADERS.items():
        if header_lower in aliases:
            return canonical
    return header_text.strip().title()


def split_into_sections(text: str) -> list[ResumeSection]:
    """
    Split raw resume text into labeled sections based on detected headers.

    If no headers are detected at all, the whole document is returned as a
    single "General" section so downstream chunking still works.
    """
    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        return [ResumeSection(section="General", content=text.strip())] if text.strip() else []

    sections: list[ResumeSection] = []

    # Anything before the first detected header (name, contact info, etc.)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(ResumeSection(section="Contact", content=preamble))

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        sections.append(
            ResumeSection(section=_canonical_section(match.group(1)), content=content)
        )

    return sections


def chunk_sections(sections: list[ResumeSection]) -> list[Document]:
    """
    Run each ResumeSection through RecursiveCharacterTextSplitter and return
    LangChain Documents tagged with their originating section name.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
    for section in sections:
        chunks = splitter.split_text(section.content)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"section": section.section, "chunk_index": chunk_idx},
                )
            )

    return documents


def chunk_resume_text(text: str) -> tuple[list[ResumeSection], list[Document]]:
    """Convenience wrapper: detect sections, then chunk them."""
    sections = split_into_sections(text)
    documents = chunk_sections(sections)
    return sections, documents
