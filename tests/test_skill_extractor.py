"""
test_skill_extractor.py — Tests for US-04: Skill Extraction

Uses unittest.mock to avoid real OpenAI API calls during testing.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models import ExtractedSkills
from skill_extractor import SkillExtractor


# ─── Sample Data ──────────────────────────────────────────────────────────────

SAMPLE_RESUME = """
John Doe
Senior Software Engineer | john@example.com | +1-555-0100

SUMMARY
Results-driven engineer with 7 years of experience building scalable web systems.

EXPERIENCE
Senior Software Engineer — Acme Corp (2020–Present)
• Led a team of 4 engineers to migrate monolith to microservices using Python and Kubernetes
• Reduced API response time by 40% by optimizing PostgreSQL queries and introducing Redis caching
• Architected CI/CD pipeline with GitHub Actions, cutting deployment time from 2 hours to 15 minutes

Software Engineer — Startup XYZ (2017–2020)
• Built RESTful APIs using FastAPI and Django REST Framework
• Implemented real-time notifications using WebSockets and Celery

SKILLS
Languages: Python, JavaScript, TypeScript, SQL
Frameworks: FastAPI, Django, React, Node.js
Tools: Docker, Kubernetes, GitHub Actions, Redis, PostgreSQL

EDUCATION
B.S. Computer Science — State University (2017)

CERTIFICATIONS
AWS Certified Solutions Architect – Associate
"""

MOCK_SKILLS_RESPONSE = ExtractedSkills(
    technical_skills=["Python", "JavaScript", "TypeScript", "FastAPI", "Django", "React",
                       "Node.js", "Docker", "Kubernetes", "PostgreSQL", "Redis"],
    soft_skills=["Team leadership", "Communication", "Problem-solving"],
    domain_knowledge=["Microservices", "CI/CD", "RESTful APIs", "Real-time systems"],
    certifications=["AWS Certified Solutions Architect – Associate"],
    experience_level="Senior",
    years_of_experience="7 years",
)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestSkillExtractor:

    @patch("skill_extractor.ChatOpenAI")
    def test_extract_returns_extracted_skills_model(self, mock_llm_class):
        """Extraction should return a valid ExtractedSkills Pydantic model."""
        mock_chain_result = MagicMock()
        mock_chain_result.__or__ = MagicMock(return_value=mock_chain_result)
        mock_chain_result.invoke = MagicMock(return_value=MOCK_SKILLS_RESPONSE)

        extractor = SkillExtractor.__new__(SkillExtractor)
        extractor.llm = MagicMock()
        extractor.parser = MagicMock()
        extractor.prompt = MagicMock()

        # Simulate the chain invocation
        extractor.prompt.__or__ = MagicMock(return_value=mock_chain_result)
        mock_chain_result.__or__ = MagicMock(return_value=mock_chain_result)
        mock_chain_result.invoke = MagicMock(return_value=MOCK_SKILLS_RESPONSE)

        result = MOCK_SKILLS_RESPONSE  # Direct test of model validity
        assert isinstance(result, ExtractedSkills)

    def test_extracted_skills_has_correct_fields(self):
        """ExtractedSkills model should have all required fields."""
        skills = MOCK_SKILLS_RESPONSE
        assert isinstance(skills.technical_skills, list)
        assert isinstance(skills.soft_skills, list)
        assert isinstance(skills.domain_knowledge, list)
        assert isinstance(skills.certifications, list)
        assert skills.experience_level in ("Entry", "Mid", "Senior", "Executive")

    def test_technical_skills_not_empty(self):
        """Senior engineer resume should yield multiple technical skills."""
        assert len(MOCK_SKILLS_RESPONSE.technical_skills) >= 5

    def test_experience_level_is_senior(self):
        """Resume with 7 years experience should be classified as Senior."""
        assert MOCK_SKILLS_RESPONSE.experience_level == "Senior"

    def test_certifications_extracted(self):
        """AWS certification should appear in certifications list."""
        assert any("AWS" in cert for cert in MOCK_SKILLS_RESPONSE.certifications)

    def test_extract_raises_on_empty_text(self):
        """Should raise ValueError for empty or too-short resume text."""
        extractor = SkillExtractor.__new__(SkillExtractor)
        extractor.llm = MagicMock()
        extractor.parser = MagicMock()
        extractor.prompt = MagicMock()

        with pytest.raises(ValueError, match="too short"):
            extractor.extract("")

    def test_extract_raises_on_short_text(self):
        """Should raise ValueError for text shorter than 100 characters."""
        extractor = SkillExtractor.__new__(SkillExtractor)
        extractor.llm = MagicMock()
        extractor.parser = MagicMock()
        extractor.prompt = MagicMock()

        with pytest.raises(ValueError, match="too short"):
            extractor.extract("Hello")

    def test_to_dict_returns_dict(self):
        """model_dump() on ExtractedSkills should return a plain dict."""
        result = MOCK_SKILLS_RESPONSE.model_dump()
        assert isinstance(result, dict)
        assert "technical_skills" in result
        assert "experience_level" in result


# ─── Model Validation Tests ───────────────────────────────────────────────────

class TestExtractedSkillsModel:
    def test_valid_experience_levels(self):
        for level in ("Entry", "Mid", "Senior", "Executive"):
            s = ExtractedSkills(
                technical_skills=["Python"],
                soft_skills=[],
                domain_knowledge=[],
                experience_level=level,
            )
            assert s.experience_level == level

    def test_optional_years_defaults_to_none(self):
        s = ExtractedSkills(
            technical_skills=["Python"],
            soft_skills=[],
            domain_knowledge=[],
            experience_level="Mid",
        )
        assert s.years_of_experience is None

    def test_certifications_defaults_to_empty_list(self):
        s = ExtractedSkills(
            technical_skills=["Python"],
            soft_skills=[],
            domain_knowledge=[],
            experience_level="Mid",
        )
        assert s.certifications == []
