"""
test_job_matcher.py — Tests for US-05: Job Description Matching

Uses mock data to validate matching logic and model structure.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models import JobMatchResult, SkillMatch
from job_matcher import JobMatcher


# ─── Sample Data ──────────────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """
John Doe | Python Developer
7 years experience with Python, Django, FastAPI, PostgreSQL, Docker, AWS.
Led teams, built microservices, deployed on Kubernetes.
"""

SAMPLE_JD = """
We are looking for a Senior Python Engineer to join our backend team.

Requirements:
- 5+ years of Python development experience
- Strong knowledge of Django or FastAPI
- Experience with PostgreSQL and Redis
- Docker and Kubernetes proficiency
- AWS experience (EC2, RDS, S3)

Nice to have:
- GraphQL experience
- TypeScript knowledge
"""

MOCK_MATCH_RESULT = JobMatchResult(
    match_score=82.5,
    matched_skills=["Python", "Django", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS"],
    missing_skills=["Redis", "GraphQL"],
    skill_breakdown=[
        SkillMatch(skill="Python", found_in_resume=True, importance="Required"),
        SkillMatch(skill="Django", found_in_resume=True, importance="Required"),
        SkillMatch(skill="FastAPI", found_in_resume=True, importance="Required"),
        SkillMatch(skill="PostgreSQL", found_in_resume=True, importance="Required"),
        SkillMatch(skill="Redis", found_in_resume=False, importance="Required"),
        SkillMatch(skill="Docker", found_in_resume=True, importance="Required"),
        SkillMatch(skill="Kubernetes", found_in_resume=True, importance="Required"),
        SkillMatch(skill="AWS", found_in_resume=True, importance="Required"),
        SkillMatch(skill="GraphQL", found_in_resume=False, importance="Nice-to-have"),
        SkillMatch(skill="TypeScript", found_in_resume=False, importance="Nice-to-have"),
    ],
    summary=(
        "Strong candidate with 7 years of Python experience and proficiency in "
        "Django, FastAPI, Docker, and AWS. Missing Redis and GraphQL, which are "
        "listed as required/preferred in the job description."
    ),
)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestJobMatchResult:
    def test_match_score_within_bounds(self):
        assert 0 <= MOCK_MATCH_RESULT.match_score <= 100

    def test_matched_skills_is_list(self):
        assert isinstance(MOCK_MATCH_RESULT.matched_skills, list)

    def test_missing_skills_is_list(self):
        assert isinstance(MOCK_MATCH_RESULT.missing_skills, list)

    def test_skill_breakdown_contains_skill_match_objects(self):
        for item in MOCK_MATCH_RESULT.skill_breakdown:
            assert isinstance(item, SkillMatch)
            assert item.importance in ("Required", "Preferred", "Nice-to-have")

    def test_matched_skills_not_in_missing(self):
        """A skill cannot be both matched and missing."""
        overlap = set(MOCK_MATCH_RESULT.matched_skills) & set(MOCK_MATCH_RESULT.missing_skills)
        assert len(overlap) == 0, f"Skills appear in both lists: {overlap}"

    def test_summary_is_non_empty_string(self):
        assert isinstance(MOCK_MATCH_RESULT.summary, str)
        assert len(MOCK_MATCH_RESULT.summary) > 20

    def test_good_match_score_reflects_mostly_found_skills(self):
        """With 7/10 skills found, score should be above 70."""
        assert MOCK_MATCH_RESULT.match_score >= 70


class TestSkillMatchModel:
    def test_importance_values(self):
        valid = {"Required", "Preferred", "Nice-to-have"}
        for match in MOCK_MATCH_RESULT.skill_breakdown:
            assert match.importance in valid

    def test_found_in_resume_is_bool(self):
        for match in MOCK_MATCH_RESULT.skill_breakdown:
            assert isinstance(match.found_in_resume, bool)


class TestJobMatcherValidation:
    def test_raises_on_empty_resume(self):
        matcher = JobMatcher.__new__(JobMatcher)
        matcher.llm = MagicMock()
        matcher.parser = MagicMock()
        matcher.prompt = MagicMock()

        with pytest.raises(ValueError):
            matcher.match("", SAMPLE_JD)

    def test_raises_on_empty_jd(self):
        matcher = JobMatcher.__new__(JobMatcher)
        matcher.llm = MagicMock()
        matcher.parser = MagicMock()
        matcher.prompt = MagicMock()

        with pytest.raises(ValueError):
            matcher.match(SAMPLE_RESUME_TEXT, "")

    def test_model_serialization(self):
        """JobMatchResult should serialize cleanly to a dict."""
        d = MOCK_MATCH_RESULT.model_dump()
        assert "match_score" in d
        assert "skill_breakdown" in d
        assert isinstance(d["skill_breakdown"], list)
