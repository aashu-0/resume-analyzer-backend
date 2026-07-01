# Resume <-> JD Analyzer — Backend

A FastAPI + LangChain RAG backend that scores a resume against a job
description and returns an overall fit score, section-level breakdowns,
missing skills, and AI-generated rewrite tips.

## Architecture -> Code mapping

| Diagram box | File |
|---|---|
| Resume upload / Job description (INPUT) | `app/routers/analyze.py` |
| Document parser (pdfplumber / python-docx) | `app/services/document_parser.py` |
| Section chunker (Skills, Exp, Education...) | `app/services/chunker.py` |
| JD preprocessor (extract skills & keywords) | `app/services/jd_preprocessor.py` |
| Embedding model (Cohere) | `app/services/embeddings.py` |
| Vector store (Pinecone / FAISS) | `app/services/vectorstore.py` |
| Retriever (top-k semantic search) | `app/services/retriever.py` |
| Semantic score (cosine similarity) | `app/services/scoring.py::compute_semantic_score` |
| Keyword match (skill overlap %) | `app/services/scoring.py::compute_keyword_match` |
| LLM judge (qualitative gap analysis) | `app/services/llm_judge.py` |
| Score report + recommendations (OUTPUT) | `app/services/pipeline.py` -> `AnalyzeResponse` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in COHERE_API_KEY and GOOGLE_API_KEY at minimum.
```

### Vector store options

- **FAISS (default)** — no extra setup, indexes are written to
  `FAISS_INDEX_DIR` and cleaned up after each analysis.
- **Pinecone** — set `VECTOR_STORE=pinecone` and fill in `PINECONE_API_KEY`,
  `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`. The index is
  created automatically (1024 dims, cosine) if it doesn't exist, and each
  analysis uses its own namespace which is deleted afterwards.

### LLM judge

The qualitative gap analysis uses Gemini 2.0 Flash via
`langchain-google-genai`. If `GOOGLE_API_KEY` is not set, the pipeline still
runs and returns semantic + keyword scores, with a placeholder
`gap_analysis`.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## API

### `POST /api/analyze`

Multipart form:

| Field | Type | Required | Notes |
|---|---|---|---|
| `resume` | file | yes | `.pdf`, `.docx`, `.txt`, or `.md` |
| `jd_text` | string | one of `jd_text`/`jd_url` | Pasted job description text |
| `jd_url` | string | one of `jd_text`/`jd_url` | URL of a job posting to fetch |

Example with `curl`:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "resume=@/path/to/resume.pdf" \
  -F "jd_text=We are looking for a Backend Engineer with FastAPI, LangChain, Pinecone and RAG experience..."
```

Response shape (`AnalyzeResponse`):

```json
{
  "resume_id": "....",
  "sections_detected": ["Contact", "Summary", "Skills", "Projects", "Experience", "Education"],
  "jd_keywords": {
    "required_skills": ["fastapi", "langchain", "pinecone", "rag"],
    "nice_to_have_skills": ["docker", "kubernetes"],
    "keywords": ["backend", "scalability"],
    "raw_text": "..."
  },
  "report": {
    "overall_score": 78.4,
    "semantic_score": {
      "overall_similarity": 81.2,
      "section_similarities": {
        "Skills": 88.0,
        "Projects": 80.5,
        "Experience": 76.3
      }
    },
    "keyword_match": {
      "overlap_percent": 75.0,
      "matched_skills": ["fastapi", "langchain", "rag"],
      "missing_skills": ["pinecone"]
    },
    "gap_analysis": {
      "fit_summary": "...",
      "strengths": ["..."],
      "gaps": ["..."],
      "rewrite_suggestions": ["..."]
    },
    "section_scores": { "...": 0.0 },
    "missing_skills": ["pinecone"],
    "rewrite_tips": ["..."]
  }
}
```

### `GET /health`

Simple health check, also reports which vector store backend is active.

## Notes

- Per the LangChain RAG skill, the same Cohere embedding model/config
  (`COHERE_EMBEDDING_MODEL`, default `embed-english-v3.0`, 1024 dims) is used
  for **both** indexing and querying — see `app/services/embeddings.py`.
- Chunking uses `RecursiveCharacterTextSplitter` with `CHUNK_SIZE=500` /
  `CHUNK_OVERLAP=50` (within the recommended 500-1500 range with ~10-20%
  overlap).
- Each analysis run gets its own isolated vector store namespace/index
  (`resume_id`), which is cleaned up after scoring to avoid leaking
  embeddings between users.
