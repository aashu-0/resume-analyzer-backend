"""
Document Parser
================
Responsible for turning an uploaded resume file
(PDF, DOCX, or plain text) into raw text that can be handed to the
Section Chunker.
"""

from __future__ import annotations

import io
from pathlib import Path

import docx
import pdfplumber

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFileTypeError(ValueError):
    pass


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber, page by page."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract text from a DOCX using python-docx, preserving paragraph order."""
    document = docx.Document(io.BytesIO(file_bytes))

    parts: list[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # Tables often hold skills / project info in resume templates.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def _extract_plain_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def parse_resume(file_bytes: bytes, filename: str) -> str:
    """
    Parse an uploaded resume file into raw text.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to determine the parser via extension).

    Returns:
        Extracted plain text.

    Raises:
        UnsupportedFileTypeError: If the file extension isn't supported.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf_text(file_bytes)
    elif ext == ".docx":
        text = _extract_docx_text(file_bytes)
    elif ext in {".txt", ".md"}:
        text = _extract_plain_text(file_bytes)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Normalize whitespace a little so downstream regex-based section
    # detection behaves consistently.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()
