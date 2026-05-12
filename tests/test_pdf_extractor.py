"""
test_pdf_extractor.py — Tests for US-03: PDF Text Extraction

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pdf_extractor import (
    PDFExtractionError,
    _clean_text,
    extract_text_from_bytes,
    get_page_count,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_minimal_pdf(content: str = "Hello World") -> bytes:
    """
    Generate a minimal valid PDF in memory (no external files needed).
    Uses raw PDF syntax — good enough for testing extraction logic.
    """
    # We'll use pypdf to generate a simple in-memory PDF
    import io
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, ArrayObject, NumberObject

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    # Add text via annotations (simplest way without reportlab)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ─── Text Cleaning Tests ───────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_null_bytes(self):
        result = _clean_text("Hello\x00World")
        assert "\x00" not in result
        assert "Hello" in result

    def test_collapses_multiple_blank_lines(self):
        result = _clean_text("Line 1\n\n\n\n\nLine 2")
        assert result.count("\n") <= 3

    def test_strips_single_char_lines(self):
        result = _clean_text("Name\na\nExperience")
        lines = [l for l in result.split("\n") if l.strip()]
        single_chars = [l for l in lines if len(l.strip()) == 1 and l.strip() not in ("•", "-")]
        assert len(single_chars) == 0

    def test_preserves_bullet_chars(self):
        result = _clean_text("Skills\n•\nPython")
        assert "•" in result

    def test_strips_leading_trailing_whitespace(self):
        result = _clean_text("   Hello World   ")
        assert result == result.strip()

    def test_handles_empty_string(self):
        result = _clean_text("")
        assert result == ""

    def test_form_feed_converted_to_newline(self):
        result = _clean_text("Page 1\x0cPage 2")
        assert "Page 1" in result
        assert "Page 2" in result
        assert "\x0c" not in result


# ─── Extraction Validation Tests ──────────────────────────────────────────────

class TestExtractTextFromBytes:
    def test_rejects_empty_bytes(self):
        with pytest.raises((PDFExtractionError, Exception)):
            extract_text_from_bytes(b"")

    def test_rejects_non_pdf_bytes(self):
        with pytest.raises(Exception):
            extract_text_from_bytes(b"This is not a PDF file at all.")

    def test_rejects_corrupt_pdf(self):
        corrupt = b"%PDF-1.4 CORRUPT DATA GOES HERE"
        with pytest.raises(Exception):
            extract_text_from_bytes(corrupt)

    def test_returns_string_type(self, tmp_path):
        """Create a real PDF with pdfplumber-readable text and test extraction."""
        pytest.importorskip("reportlab")  # skip if reportlab not installed
        from reportlab.pdfgen import canvas

        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "John Doe")
        c.drawString(100, 730, "Python Developer")
        c.drawString(100, 710, "Skills: Python, Django, PostgreSQL")
        c.save()

        result = extract_text_from_bytes(pdf_path.read_bytes())
        assert isinstance(result, str)
        assert len(result) > 10


# ─── Page Count Tests ─────────────────────────────────────────────────────────

class TestGetPageCount:
    def test_returns_zero_for_corrupt_bytes(self):
        result = get_page_count(b"not a pdf")
        assert result == 0

    def test_returns_integer(self, tmp_path):
        pytest.importorskip("reportlab")
        from reportlab.pdfgen import canvas

        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Page 1")
        c.showPage()
        c.drawString(100, 750, "Page 2")
        c.save()

        count = get_page_count(pdf_path.read_bytes())
        assert count == 2
