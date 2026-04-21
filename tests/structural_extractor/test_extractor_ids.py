from pathlib import Path
from src.services.structural_extractor.extractors.ids import extract_ids
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_extract_invoice_id_from_real_pdf():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_ids(doc, doc_type="Invoice")
    assert "invoice_id" in out
    assert out["invoice_id"].value == "INV600254"
    assert out["invoice_id"].provenance == "extracted"
    assert out["invoice_id"].anchor_ref is not None
