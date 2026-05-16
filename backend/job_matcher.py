"""
job_matcher.py — US-05: Job Description Matching

Semantically matches a resume against a job description using LangChain.
Returns a structured match score, matched/missing skills, and a summary.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from models import JobMatchResult

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ─── Prompt Engineering ───────────────────────────────────────────────────────

JOB_MATCH_PROMPT = """\
You are an expert technical recruiter performing a resume-to-job-description match analysis.

Given a candidate's resume and a job description, provide a detailed match analysis.

{format_instructions}

SCORING GUIDELINES:
- 90-100: Excellent match — candidate meets nearly all requirements
- 70-89 : Good match — candidate meets most key requirements with minor gaps
- 50-69 : Partial match — candidate has foundational skills but notable gaps
- 30-49 : Weak match — significant skill gaps, would need substantial upskilling
- 0-29  : Poor match — candidate's background doesn't align with the role

RULES:
- Base match_score purely on skill/experience overlap, not writing quality
- List ALL required/preferred skills from the JD in skill_breakdown
- Mark found_in_resume=true only if the skill is clearly present in the resume
- Be specific in the summary — mention actual skills and gaps by name

---
JOB DESCRIPTION:
{job_description}

---
RESUME TEXT:
{resume_text}
---

Respond ONLY with the JSON object. No preamble or explanation.
"""


# ─── Matcher Class ────────────────────────────────────────────────────────────

class JobMatcher:
    """
    Matches a resume against a job description using an LLM chain.
    
    Returns structured match data: score, matched skills, gaps, and summary.
    """

    def __init__(self, model: str = None, temperature: float = None):
        model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.parser = PydanticOutputParser(pydantic_object=JobMatchResult)
        self.prompt = PromptTemplate(
            template=JOB_MATCH_PROMPT,
            input_variables=["resume_text", "job_description"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def match(self, resume_text: str, job_description: str) -> JobMatchResult:
        """
        Perform resume-to-job-description matching.
        
        Args:
            resume_text: Extracted text from the candidate's resume PDF.
            job_description: Full text of the job description.
            
        Returns:
            JobMatchResult with score, matched skills, gaps, and summary.
        """
        if not resume_text or not job_description:
            raise ValueError("Both resume_text and job_description are required.")

        # Truncate inputs to avoid exceeding context limits
        resume_trimmed = resume_text[:6000]
        jd_trimmed = job_description[:3000]

        try:
            import re
            chain = self.prompt | self.llm
            raw = chain.invoke({"resume_text": resume_trimmed, "job_description": jd_trimmed})
            text = raw.content if hasattr(raw, 'content') else str(raw)
            text = re.sub(r'^```(?:json)?\s*', '', text.strip())
            text = re.sub(r'\s*```$', '', text)
            result: JobMatchResult = self.parser.parse(text)
            logger.info(
                f"Job match score: {result.match_score}/100, "
                f"{len(result.matched_skills)} matched, "
                f"{len(result.missing_skills)} missing"
            )
            return result

        except Exception as e:
            logger.error(f"Job matching failed: {e}")
            raise ValueError(f"Job matching failed: {str(e)}") from e


# ─── Module-level convenience function ───────────────────────────────────────

_matcher_singleton: JobMatcher | None = None


def get_matcher() -> JobMatcher:
    global _matcher_singleton
    if _matcher_singleton is None:
        _matcher_singleton = JobMatcher()
    return _matcher_singleton


def match_resume_to_job(resume_text: str, job_description: str) -> JobMatchResult:
    """Top-level function for matching resume to a job description."""
    return get_matcher().match(resume_text, job_description)
