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
