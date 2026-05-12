"""
pdf_extractor.py — US-03: PDF Text Extraction

Extracts clean, structured text from uploaded resume PDFs.
Uses pdfplumber as primary extractor (better for multi-column layouts)
with pypdf as a fallback.
"""

import io
import logging
from pathlib import Path
from typing import Union

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed or yields no text."""
    pass


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF given its raw bytes.
    
    Tries pdfplumber first (better layout handling), falls back to pypdf.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file.
        
    Returns:
        Extracted text as a single string.
        
    Raises:
        PDFExtractionError: If extraction fails or the PDF has no extractable text.
    """
    text = _extract_with_pdfplumber(pdf_bytes)

    # Fallback to pypdf if pdfplumber returns nothing useful
    if not text or len(text.strip()) < 50:
        logger.warning("pdfplumber returned little text — trying pypdf fallback")
        text = _extract_with_pypdf(pdf_bytes)

    if not text or len(text.strip()) < 50:
        raise PDFExtractionError(
            "Could not extract meaningful text from the PDF. "
            "The file may be scanned/image-based. Please upload a text-based PDF."
        )

    return _clean_text(text)


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Convenience wrapper: extract text from a PDF file path.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        Extracted and cleaned text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    return extract_text_from_bytes(path.read_bytes())


# ─── Private helpers ──────────────────────────────────────────────────────────

def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Use pdfplumber for layout-aware extraction."""
    try:
        pages_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if page_text:
                    pages_text.append(f"[Page {i + 1}]\n{page_text}")
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
        return ""


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    """Fallback: use pypdf for simpler text extraction."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")
        return ""


def _clean_text(text: str) -> str:
    """
    Clean extracted text: normalize whitespace, remove junk characters,
    preserve meaningful line breaks.
    """
    import re

    # Remove null bytes and form feeds
    text = text.replace("\x00", "").replace("\x0c", "\n")

    # Collapse multiple blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just noise (single chars, page numbers, etc.)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines with meaningful content
        if len(stripped) > 1 or stripped in ("•", "-", "–", "—"):
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def get_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return 0
