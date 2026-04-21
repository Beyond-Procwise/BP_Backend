from pathlib import Path
from src.services.structural_extractor.extractors.ids import extract_ids
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIXDIR = Path(__file__).parent / "fixtures/docs"
FIX = FIXDIR / "INV600254.pdf"


def test_extract_invoice_id_from_real_pdf():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_ids(doc, doc_type="Invoice")
    assert "invoice_id" in out
    assert out["invoice_id"].value == "INV600254"
    assert out["invoice_id"].provenance == "extracted"
    assert out["invoice_id"].anchor_ref is not None


def test_inv600254_po_id_not_bespoke_fragment():
    """po_id should be PO502004, not '3-5' from 'Bespoke' substring match."""
    fixture = FIXDIR / "INV600254.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_ids(doc, doc_type="Invoice")
    assert "po_id" in out
    assert out["po_id"].value == "PO502004"


def test_dha_invoice_id_not_po_reference():
    """DHA invoice_id should be DHA-2025-143, not PO438295."""
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_ids(doc, doc_type="Invoice")
    assert out.get("invoice_id") is not None
    assert out["invoice_id"].value == "DHA-2025-143"


def test_dha_po_id_not_swift_code():
    """DHA po_id should be PO438295, not a SWIFT code like METRGB2L."""
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_ids(doc, doc_type="Invoice")
    assert out.get("po_id") is not None
    # SWIFT/BIC codes are 8 or 11 alphanumeric with bank prefix — reject.
    assert out["po_id"].value != "METRGB2L"
    assert out["po_id"].value == "PO438295"
