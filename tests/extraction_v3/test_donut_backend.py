"""Tests for the Donut OCR-free backend adapter.

GPU-gated via @pytest.mark.gpu — deselect on CPU-only runners with:
    pytest -m "not gpu"
"""
from pathlib import Path
import pytest
from src.services.extraction_v3.parsers.donut_backend import parse_with_donut

FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_donut_parses_scanned_image_pdf():
    """Donut should extract SOMETHING from the scanned-PDF fixture; even pre-trained
    base, it produces structured-text output. Test asserts non-empty result."""
    doc = parse_with_donut(FX / "INV-005-scanned.pdf", file_format="pdf-scanned")
    assert doc.parser_backend == "donut"
    assert doc.full_text  # non-empty
    # Substring guarantee proxy: every token's text must be in full_text
    for page in doc.pages:
        for tok in page.tokens:
            assert tok.text in doc.full_text, (
                f"token {tok.text!r} not in full_text — substring guarantee broken"
            )
    assert doc.parser_confidence > 0
