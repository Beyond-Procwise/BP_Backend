from pathlib import Path
from src.services.structural_extractor.extractors.parties import extract_parties
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIXDIR = Path(__file__).parent / "fixtures/docs"
FIX = FIXDIR / "INV600254.pdf"


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


# === F4: supplier/buyer extraction across docs ===

def test_dha_parties_all_caps_letterhead():
    """DHA: 'DESIGN HOUSE AGENCY' (no corp suffix) is supplier;
    'ASSURITY LTD' (uppercase LTD) is buyer from BILL TO anchor."""
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DHA fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_parties(doc, "Invoice")
    assert out.get("supplier_id") is not None
    assert "DESIGN HOUSE AGENCY" in out["supplier_id"].value.upper()
    # Buyer from BILL TO
    assert out.get("buyer_id") is not None
    assert "ASSURITY" in out["buyer_id"].value.upper()


def test_eleanor_supplier_multiline_letterhead():
    """ELEANOR: two-line letterhead 'ELEANOR PRICE' + 'CREATIVE STUDIO'
    must be merged into a single supplier name."""
    fixture = FIXDIR / "ELEANOR-2025-290.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("ELEANOR fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_parties(doc, "Invoice")
    assert out.get("supplier_id") is not None
    sup = out["supplier_id"].value.lower()
    assert "eleanor" in sup and "price" in sup and "creative" in sup and "studio" in sup


def test_eleanor_buyer_from_billed_to():
    """ELEANOR buyer 'Assurity Ltd' via 'Billed to:' anchor."""
    fixture = FIXDIR / "ELEANOR-2025-290.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("ELEANOR fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_parties(doc, "Invoice")
    assert out.get("buyer_id") is not None
    assert "assurity" in out["buyer_id"].value.lower()


def test_duncan_supplier_via_recipient_anchor():
    """DUNCAN PO: 'Recipient:' is the supplier anchor (the party receiving
    the PO). 'Duncan LLC' should be the supplier_name."""
    fixture = FIXDIR / "DUNCAN_PO526800.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DUNCAN fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_parties(doc, "Purchase_Order")
    # supplier_name or supplier_id — schema may use either
    val = None
    if "supplier_name" in out:
        val = out["supplier_name"].value
    elif "supplier_id" in out:
        val = out["supplier_id"].value
    assert val is not None, f"no supplier found: {out}"
    assert "duncan" in val.lower()
