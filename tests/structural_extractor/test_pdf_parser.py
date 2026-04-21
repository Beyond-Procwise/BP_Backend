from pathlib import Path
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.parsing.model import BBox, Token

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_parse_pdf_produces_tokens():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture not available")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    assert doc.source_format == "pdf"
    assert doc.pages_or_sheets == 1
    assert len(doc.tokens) > 20
    for t in doc.tokens:
        assert isinstance(t.anchor, BBox)
        assert t.anchor.page == 1


def test_parse_pdf_full_text_contains_key_tokens():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture not available")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    assert "INV600254" in doc.full_text
    assert "City of Newport" in doc.full_text
