"""
models.py — Pydantic data models for structured LLM output parsing.

These models define the exact schema that the LLM must return.
LangChain's PydanticOutputParser uses these to validate and parse responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ─── US-04: Skill Extraction Output ───────────────────────────────────────────

class ExtractedSkills(BaseModel):
    """Structured output from the skill extraction LLM chain."""

    technical_skills: List[str] = Field(
        description="Programming languages, frameworks, tools, and technologies",
        examples=[["Python", "React", "PostgreSQL", "Docker"]],
    )
    soft_skills: List[str] = Field(
        description="Interpersonal and professional skills",
        examples=[["Team leadership", "Communication", "Problem-solving"]],
    )
    domain_knowledge: List[str] = Field(
        description="Industry or domain-specific knowledge areas",
        examples=[["Machine Learning", "Financial modeling", "Healthcare compliance"]],
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Professional certifications and licenses",
        examples=[["AWS Certified Solutions Architect", "PMP"]],
    )
    experience_level: str = Field(
        description="Detected seniority level: 'Entry', 'Mid', 'Senior', or 'Executive'",
        examples=["Senior"],
    )
    years_of_experience: Optional[str] = Field(
        default=None,
        description="Estimated or explicitly stated years of experience",
        examples=["5-7 years"],
    )


# ─── US-05: Job Description Matching Output ───────────────────────────────────

class SkillMatch(BaseModel):
    """A single matched or missing skill between resume and job description."""

    skill: str = Field(description="The skill or keyword")
    found_in_resume: bool = Field(description="Whether this skill appears in the resume")
    importance: str = Field(
        description="How critical this skill is for the role: 'Required', 'Preferred', or 'Nice-to-have'"
    )


class JobMatchResult(BaseModel):
    """Structured output from the job description matching chain."""

    match_score: float = Field(
        description="Overall match percentage between resume and job description (0-100)",
        ge=0,
        le=100,
    )
    matched_skills: List[str] = Field(
        description="Skills present in both the resume and the job description"
    )
    missing_skills: List[str] = Field(
        description="Required or preferred skills from the job description absent from the resume"
    )
    skill_breakdown: List[SkillMatch] = Field(
        description="Detailed breakdown of each skill from the job description"
    )
    summary: str = Field(
        description="2-3 sentence plain-English summary of how well the candidate fits the role"
    )


# ─── ATS Scoring Output ───────────────────────────────────────────────────────

class ATSCriterion(BaseModel):
    """A single ATS evaluation criterion with score and feedback."""

    criterion: str = Field(description="What is being evaluated, e.g., 'Keyword Density'")
    score: int = Field(description="Score for this criterion out of 100", ge=0, le=100)
    feedback: str = Field(description="Specific, actionable feedback for this criterion")


class ATSScoreResult(BaseModel):
    """Full ATS analysis of a resume."""

    overall_score: int = Field(
        description="Overall ATS score out of 100", ge=0, le=100
    )
    criteria_scores: List[ATSCriterion] = Field(
        description="Breakdown of scores across individual ATS criteria"
    )
    top_improvements: List[str] = Field(
        description="Top 3-5 most impactful improvements the candidate should make",
        max_length=5,
    )
    strengths: List[str] = Field(
        description="2-3 things the resume does well"
    )
    verdict: str = Field(
        description="One-line verdict, e.g., 'Strong candidate — minor keyword gaps only'"
    )


# ─── Combined Full Analysis Response ──────────────────────────────────────────

class FullAnalysisResult(BaseModel):
    """Top-level response returned by the /analyze endpoint."""

    filename: str
    extracted_text_length: int = Field(description="Number of characters extracted from the PDF")
    skills: ExtractedSkills
    ats_score: ATSScoreResult
    job_match: Optional[JobMatchResult] = Field(
        default=None,
        description="Populated only when a job description is provided"
    )
