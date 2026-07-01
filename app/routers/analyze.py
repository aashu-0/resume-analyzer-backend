from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas import AnalyzeResponse
from app.services.document_parser import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError
from app.services.pipeline import analyze_resume

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile,
    jd_text: str,
) -> AnalyzeResponse:
    """
    Run the full Resume <-> Job Description analysis pipeline.

    - **resume**: PDF, DOCX, or plain-text resume file.
    - **jd_text**: Job description pasted as plain text.
    - **jd_url**: Alternatively, a URL to a job posting (will be fetched and
      cleaned). At least one of `jd_text` or `jd_url` is required.
    """

    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty resume file.")

    try:
        return analyze_resume(resume_bytes, resume.filename or "resume", jd_text)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{exc} Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
