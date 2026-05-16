"""
main.py — FastAPI Application

REST API for the AI Resume Analyzer.
Exposes endpoints for US-02 (PDF upload), US-03 (extraction),
US-04 (skill extraction), US-05 (job matching), and ATS scoring.

Run with:
    uvicorn main:app --reload --port 8000
"""

import logging
import requests
import os
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ats_scorer import score_resume
from job_matcher import match_resume_to_job
from models import ATSScoreResult, ExtractedSkills, FullAnalysisResult, JobMatchResult
from pdf_extractor import PDFExtractionError, extract_text_from_bytes, get_page_count
from skill_extractor import extract_skills

load_dotenv(override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Resume Analyzer",
    description=(
        "Upload a resume PDF to extract skills, match job descriptions, "
        "get an ATS score, and receive improvement suggestions."
    ),
    version="1.0.0",
)

# Allow the frontend (running on any port locally) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _read_and_validate_pdf(file: UploadFile) -> bytes:
    """Read uploaded file, validate type and size, return raw bytes."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF files are accepted. Got: {file.content_type}",
            )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return pdf_bytes


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running and the Groq API key is configured."""
    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "ok",
        "api_key_configured": api_key_set,
        "message": "Ready" if api_key_set else "Warning: GROQ_API_KEY not set",
    }


@app.post("/upload-resume", tags=["US-02 Resume Upload"], response_model=dict)
async def upload_resume(file: UploadFile = File(...)):
    """
    US-02 + US-03: Upload a resume PDF and extract its text.
    
    Returns the extracted text and metadata (page count, character count).
    Use this endpoint to test PDF extraction independently of the LLM.
    """
    pdf_bytes = await _read_and_validate_pdf(file)

    try:
        extracted_text = extract_text_from_bytes(pdf_bytes)
        page_count = get_page_count(pdf_bytes)
    except PDFExtractionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during PDF extraction")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return {
        "filename": file.filename,
        "page_count": page_count,
        "character_count": len(extracted_text),
        "extracted_text": extracted_text,
        "message": "Resume extracted successfully",
    }


@app.post("/extract-skills", tags=["US-04 Skill Extraction"], response_model=ExtractedSkills)
async def extract_skills_endpoint(file: UploadFile = File(...)):
    """
    US-04: Extract structured skills from a resume PDF.
    
    Returns technical skills, soft skills, domain knowledge,
    certifications, and experience level.
    """
    pdf_bytes = await _read_and_validate_pdf(file)

    try:
        resume_text = extract_text_from_bytes(pdf_bytes)
        skills = extract_skills(resume_text)
    except PDFExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Skill extraction error")
        raise HTTPException(status_code=500, detail=str(e))

    return skills


@app.post("/match-job", tags=["US-05 Job Matching"], response_model=JobMatchResult)
async def match_job(
    file: UploadFile = File(...),
    job_description: str = Form(..., description="Full text of the job description"),
):
    """
    US-05: Match a resume against a job description.
    
    Returns a match score (0-100), matched skills, missing skills,
    and a plain-English summary.
    """
    if not job_description or len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description must be at least 50 characters.",
        )

    pdf_bytes = await _read_and_validate_pdf(file)

    try:
        resume_text = extract_text_from_bytes(pdf_bytes)
        match_result = match_resume_to_job(resume_text, job_description)
    except PDFExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Job matching error")
        raise HTTPException(status_code=500, detail=str(e))

    return match_result


@app.post("/ats-score", tags=["ATS Scoring"], response_model=ATSScoreResult)
async def ats_score(file: UploadFile = File(...)):
    """
    Score a resume on ATS (Applicant Tracking System) compatibility.
    
    Returns an overall score, criterion breakdown, strengths,
    top improvement suggestions, and a verdict.
    """
    pdf_bytes = await _read_and_validate_pdf(file)

    try:
        resume_text = extract_text_from_bytes(pdf_bytes)
        score_result = score_resume(resume_text)
    except PDFExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("ATS scoring error")
        raise HTTPException(status_code=500, detail=str(e))

    return score_result


@app.post("/analyze", tags=["Full Analysis"], response_model=FullAnalysisResult)
async def full_analysis(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(
        default=None,
        description="Optional job description to include match analysis",
    ),
):
    """
    Full pipeline: extract text → skills → ATS score → (optional) job match.
    
    This is the primary endpoint for the UI. Runs all analysis in sequence.
    Provide a job_description form field to also get match analysis.
    """
    pdf_bytes = await _read_and_validate_pdf(file)

    # Step 1: Extract text (US-03)
    try:
        resume_text = extract_text_from_bytes(pdf_bytes)
    except PDFExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 2: Extract skills (US-04)
    try:
        skills = extract_skills(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill extraction failed: {e}")

    # Step 3: ATS scoring
    try:
        ats_result = score_resume(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ATS scoring failed: {e}")

    # Step 4: Job matching (US-05) — optional
    job_match = None
    if job_description and len(job_description.strip()) >= 50:
        try:
            job_match = match_resume_to_job(resume_text, job_description)
        except Exception as e:
            logger.warning(f"Job matching failed (non-fatal): {e}")
            # Don't fail the whole request if matching fails

    return FullAnalysisResult(
        filename=file.filename,
        extracted_text_length=len(resume_text),
        skills=skills,
        ats_score=ats_result,
        job_match=job_match,
    )

@app.get("/search-jobs")
def search_jobs(role: str, location: str = "India"):

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise HTTPException(
            status_code=500,
            detail="Adzuna API keys not configured"
        )

    url = (
        f"https://api.adzuna.com/v1/api/jobs/in/search/1"
        f"?app_id={app_id}"
        f"&app_key={app_key}"
        f"&what={role}"
        f"&where={location}"
        f"&results_per_page=10"
    )

    try:
        response = requests.get(url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch jobs"
            )

        data = response.json()

        jobs = []

        for job in data.get("results", []):

            jobs.append({
                "title": job.get("title"),
                "company": job.get("company", {}).get("display_name"),
                "location": job.get("location", {}).get("display_name"),
                "description": job.get("description")
            })

        return {"jobs": jobs}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    uvicorn.run("main:app", host=host, port=port, reload=debug, log_level="info")
