from pathlib import Path
from src.services.structural_extractor.extractors.parties import extract_parties
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_extract_newport_parties():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_parties(doc, "Invoice")
    assert "supplier_id" in out
    assert "Newport" in out["supplier_id"].value
    assert "buyer_id" in out
    assert "Assurity" in out["buyer_id"].value
    assert "Ltd" in out["buyer_id"].value
