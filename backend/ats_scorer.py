"""
ats_scorer.py — ATS Score Generation & Improvement Suggestions

Evaluates a resume against ATS (Applicant Tracking System) criteria:
  - Keyword density and placement
  - Format compatibility (section headers, dates, etc.)
  - Contact information completeness
  - Quantified achievements
  - Action verb usage
  - Length and structure

This is the "real-world feature" layer on top of basic skill extraction.
"""

import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from models import ATSScoreResult

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ─── ATS Scoring Prompt ───────────────────────────────────────────────────────

ATS_SCORE_PROMPT = """\
You are an ATS (Applicant Tracking System) expert and resume coach.

Evaluate the resume below across the following criteria. Score each criterion
from 0-100 and provide specific, actionable feedback.

{format_instructions}

EVALUATION CRITERIA:
1. Keyword Density — Are relevant technical keywords prominent and repeated naturally?
2. Section Headers — Does the resume use standard ATS-readable section names?
   (e.g., "Work Experience" not "My Journey", "Education" not "Where I Studied")
3. Contact Information — Is name, email, phone, LinkedIn clearly present?
4. Quantified Achievements — Does the candidate use numbers/metrics in accomplishments?
   (e.g., "Increased revenue by 30%" not "Improved revenue")
5. Action Verbs — Does each bullet point start with a strong action verb?
   (e.g., "Developed", "Led", "Reduced", "Architected")
6. Formatting Compatibility — Is the resume free of tables, images, headers/footers,
   and complex formatting that ATS systems struggle to parse?
7. Length & Structure — Is the resume 1-2 pages with consistent date formatting?

For top_improvements, list the 3-5 changes that would have the BIGGEST impact on ATS score.
For strengths, list 2-3 things the resume genuinely does well.
For verdict, give a crisp one-line assessment.

---
RESUME TEXT:
{resume_text}
---

Respond ONLY with the JSON object. No preamble.
"""


class ATSScorer:
    """
    Evaluates a resume's ATS-readiness and suggests improvements.
    
    ATS systems are used by 98% of Fortune 500 companies to screen resumes.
    This scorer helps candidates understand if their resume will make it through.
    """

    def __init__(self, model: str = None, temperature: float = None):
        model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.parser = PydanticOutputParser(pydantic_object=ATSScoreResult)
        self.prompt = PromptTemplate(
            template=ATS_SCORE_PROMPT,
            input_variables=["resume_text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def score(self, resume_text: str) -> ATSScoreResult:
        """
        Evaluate the resume for ATS compatibility.
        
        Args:
            resume_text: Full extracted text from the resume PDF.
            
        Returns:
            ATSScoreResult with overall score, breakdown, and suggestions.
        """
        if not resume_text or len(resume_text.strip()) < 100:
            raise ValueError("Resume text is too short to score.")

        # Pre-screening heuristic: detect obvious ATS red flags in text
        _log_ats_red_flags(resume_text)

        truncated = resume_text[:7000]

        try:
            import re
            chain = self.prompt | self.llm
            raw = chain.invoke({"resume_text": truncated})
            text = raw.content if hasattr(raw, 'content') else str(raw)
            text = re.sub(r'^```(?:json)?\s*', '', text.strip())
            text = re.sub(r'\s*```$', '', text)
            result: ATSScoreResult = self.parser.parse(text)
            logger.info(
                f"ATS score: {result.overall_score}/100, "
                f"{len(result.criteria_scores)} criteria evaluated, "
                f"verdict: {result.verdict}"
            )
            return result

        except Exception as e:
            logger.error(f"ATS scoring failed: {e}")
            raise ValueError(f"ATS scoring failed: {str(e)}") from e


# ─── Heuristic Pre-screening ──────────────────────────────────────────────────

def _log_ats_red_flags(text: str) -> None:
    """
    Log potential ATS red flags detected via simple heuristics.
    These don't affect the score but help with debugging.
    """
    flags = []

    # Check for common non-standard section names
    unusual_headers = [
        r"\bmy journey\b", r"\bwhere i (studied|worked)\b",
        r"\bwhat i (know|do|did)\b", r"\bpassions?\b",
    ]
    for pattern in unusual_headers:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(f"Non-standard section header detected: {pattern}")

    # Check for missing contact info patterns
    if not re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text):
        flags.append("No email address detected")

    if not re.search(r"\+?[\d\s\-().]{10,15}", text):
        flags.append("No phone number detected")

    if flags:
        logger.debug(f"ATS pre-screening flags: {flags}")


# ─── Module-level convenience ─────────────────────────────────────────────────

_scorer_singleton: ATSScorer | None = None


def get_scorer() -> ATSScorer:
    global _scorer_singleton
    if _scorer_singleton is None:
        _scorer_singleton = ATSScorer()
    return _scorer_singleton


def score_resume(resume_text: str) -> ATSScoreResult:
    """Top-level function for ATS scoring."""
    return get_scorer().score(resume_text)
