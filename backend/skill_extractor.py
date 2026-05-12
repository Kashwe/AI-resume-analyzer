"""
skill_extractor.py — US-04: Skill Extraction via LLM

Uses LangChain with a PydanticOutputParser to extract structured skill data
from raw resume text. Demonstrates:
  - Prompt engineering for structured extraction
  - PydanticOutputParser for reliable JSON output
  - LLMChain composition
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from models import ExtractedSkills

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Prompt Engineering ───────────────────────────────────────────────────────

SKILL_EXTRACTION_PROMPT = """\
You are an expert technical recruiter and resume analyst.

Analyze the resume text below and extract structured information about the candidate.

{format_instructions}

IMPORTANT RULES:
- Extract ONLY skills explicitly mentioned or clearly implied by the resume text
- Do NOT invent or hallucinate skills not present in the text
- For experience_level, choose exactly one of: Entry, Mid, Senior, Executive
- Normalize skill names (e.g., "JS" → "JavaScript", "ML" → "Machine Learning")
- Separate tools/frameworks from the language they belong to
  (e.g., "React" is a skill separate from "JavaScript")
- For years_of_experience, state a range or exact number if computable from dates;
  otherwise return null

Resume Text:
-----------
{resume_text}
-----------

Respond ONLY with the JSON object matching the schema above. No preamble.
"""


# ─── Extractor Class ──────────────────────────────────────────────────────────

class SkillExtractor:
    """
    Extracts structured skills from resume text using LangChain + Groq.
    
    Usage:
        extractor = SkillExtractor()
        skills = extractor.extract("John has 5 years of Python experience...")
    """

    def __init__(self, model: str = None, temperature: float = None):
        model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.parser = PydanticOutputParser(pydantic_object=ExtractedSkills)
        self.prompt = PromptTemplate(
            template=SKILL_EXTRACTION_PROMPT,
            input_variables=["resume_text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def extract(self, resume_text: str) -> ExtractedSkills:
        """
        Extract skills from resume text.
        
        Args:
            resume_text: Raw text extracted from the PDF.
            
        Returns:
            ExtractedSkills — validated Pydantic model with all skill categories.
            
        Raises:
            ValueError: If the LLM returns unparseable output.
        """
        if not resume_text or len(resume_text.strip()) < 100:
            raise ValueError("Resume text is too short to extract meaningful skills.")

        # Truncate very long resumes to avoid token limits (keep first 8000 chars)
        truncated = resume_text[:8000]
        if len(resume_text) > 8000:
            logger.info(
                f"Resume truncated from {len(resume_text)} to 8000 chars for skill extraction"
            )

        try:
            import re
            chain = self.prompt | self.llm
            raw = chain.invoke({"resume_text": truncated})
            text = raw.content if hasattr(raw, 'content') else str(raw)
            text = re.sub(r'^```(?:json)?\s*', '', text.strip())
            text = re.sub(r'\s*```$', '', text)
            result: ExtractedSkills = self.parser.parse(text)
            logger.info(
                f"Extracted {len(result.technical_skills)} technical skills, "
                f"{len(result.soft_skills)} soft skills. "
                f"Experience level: {result.experience_level}"
            )
            return result

        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            raise ValueError(f"Skill extraction failed: {str(e)}") from e

    def extract_to_dict(self, resume_text: str) -> dict:
        """Convenience method — returns a plain dict instead of Pydantic model."""
        return self.extract(resume_text).model_dump()


# ─── Module-level convenience function ───────────────────────────────────────

@lru_cache(maxsize=1)
def get_extractor() -> SkillExtractor:
    """Return a cached singleton SkillExtractor (avoids repeated model init)."""
    return SkillExtractor()


def extract_skills(resume_text: str) -> ExtractedSkills:
    """Top-level function for extracting skills. Uses cached extractor."""
    return get_extractor().extract(resume_text)
