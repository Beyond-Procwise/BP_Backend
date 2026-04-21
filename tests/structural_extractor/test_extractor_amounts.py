from pathlib import Path
from src.services.structural_extractor.extractors.amounts import extract_amounts
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_extract_newport_amounts():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_amounts(doc, "Invoice")
    assert abs(out["invoice_amount"].value - 8333.0) < 0.01
    assert abs(out["tax_amount"].value - 1666.60) < 0.01
    assert abs(out["invoice_total_incl_tax"].value - 9999.60) < 0.01
    assert out["currency"].value == "GBP"
