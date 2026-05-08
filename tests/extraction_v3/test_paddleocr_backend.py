"""Tests for the PaddleOCR PP-Structure backend adapter.

GPU-gated via @pytest.mark.gpu — deselect on CPU-only runners with:
    pytest -m "not gpu"
"""

from pathlib import Path

import pytest

from src.services.extraction_v3.parsers.paddleocr_backend import parse_with_paddleocr

FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_paddleocr_parses_scanned_pdf():
    doc = parse_with_paddleocr(FX / "INV-005-scanned.pdf", file_format="pdf-scanned")
    assert doc.parser_backend == "paddleocr"
    assert doc.file_format == "pdf-scanned"
    assert len(doc.pages) >= 1
    # OCR'd a real invoice — full_text should have something meaningful
    assert doc.full_text, "PaddleOCR returned empty full_text from scanned PDF"
    # Should have detected at least some text tokens
    assert any(p.tokens for p in doc.pages), "no tokens detected on any page"
    # Should contain at least one recognisable invoice word
    assert any(
        word.lower() in doc.full_text.lower()
        for word in ["invoice", "total", "company"]
    ), "full_text does not contain any expected invoice keywords"


@pytest.mark.gpu
def test_paddleocr_substring_guarantee():
    """Every token's text must appear in full_text — Task 19 hallucination check relies on this."""
    doc = parse_with_paddleocr(FX / "INV-005-scanned.pdf", file_format="pdf-scanned")
    for page in doc.pages:
        for tok in page.tokens:
            assert tok.text in doc.full_text, (
                f"token {tok.text!r} not in full_text — substring guarantee broken"
            )
