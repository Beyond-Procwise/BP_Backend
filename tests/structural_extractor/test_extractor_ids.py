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


# === F8: filename-hint fallback for po_id when not in PDF body ===

def test_aquarius_po_id_from_filename_hint():
    """AQUARIUS invoices reference their PO only in the filename
    ('AQUARIUS INV-25-050 for PO508084 .pdf'). When the PDF body has
    no PO token, the extractor falls back to parsing the filename."""
    for name in (
        "AQUARIUS INV-25-050 for PO508084 .pdf",
        "AQUARIUS INV-25-054 for PO508084 .pdf",
    ):
        fixture = FIXDIR / name
        if not fixture.exists():
            import pytest; pytest.skip(f"{name} missing")
        doc = parse_pdf(fixture.read_bytes(), name)
        out = extract_ids(doc, doc_type="Invoice")
        assert out.get("po_id") is not None, f"{name}: po_id missing"
        assert out["po_id"].value == "PO508084", (
            f"{name}: expected PO508084, got {out['po_id'].value!r}"
        )
        assert out["po_id"].source == "filename_hint"


def test_filename_hint_does_not_override_invoice_id():
    """The filename hint must NOT fill invoice_id — the invoice always
    has its own ID in the body; filename is less authoritative."""
    fixture = FIXDIR / "AQUARIUS INV-25-050 for PO508084 .pdf"
    if not fixture.exists():
        import pytest; pytest.skip("AQUARIUS fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_ids(doc, doc_type="Invoice")
    assert out.get("invoice_id") is not None
    assert out["invoice_id"].value == "INV-25-050"
    # invoice_id must come from the body, not from filename
    assert out["invoice_id"].source != "filename_hint"


def test_filename_hint_does_not_override_existing_po_id():
    """INV600254 has PO502004 in the body. The filename hint must not
    replace a body-extracted value."""
    fixture = FIXDIR / "INV600254.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("INV600254 fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_ids(doc, doc_type="Invoice")
    assert out.get("po_id") is not None
    assert out["po_id"].value == "PO502004"
    assert out["po_id"].source != "filename_hint"
