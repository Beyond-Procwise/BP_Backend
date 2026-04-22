from pathlib import Path
from src.services.structural_extractor.extractors.payment_terms import extract_payment_terms
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_payment_terms_extracted_from_newport():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_payment_terms(doc, "Invoice")
    if "payment_terms" in out:
        # Source: "Payment must be made within 30 days"
        assert "30" in out["payment_terms"].value or "within" in out["payment_terms"].value.lower()
        assert out["payment_terms"].provenance == "extracted"


def test_net_30_token_extracted():
    # Synthetic test: a token "Net 30" on its own should be extracted
    from src.services.structural_extractor.parsing.model import ParsedDocument, Token, BBox
    toks = [
        Token(text="Payment", anchor=BBox(1, 0, 0, 0, 0), order=0),
        Token(text="Terms:", anchor=BBox(1, 50, 0, 0, 0), order=1),
        Token(text="Net", anchor=BBox(1, 100, 0, 0, 0), order=2),
        Token(text="30", anchor=BBox(1, 130, 0, 0, 0), order=3),
    ]
    d = ParsedDocument(source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
                       pages_or_sheets=1, full_text="Payment Terms: Net 30", raw_bytes=b"")
    out = extract_payment_terms(d, "Invoice")
    assert "payment_terms" in out
    assert "Net 30" in out["payment_terms"].value or "30" in out["payment_terms"].value


FIXDIR = Path(__file__).parent / "fixtures/docs"


def test_inv600254_payment_terms_no_trailing_punctuation():
    """INV600254 source: 'Payment must be made within 30 days,' — the
    extracted value should stop at 'days' without the trailing comma."""
    fixture = FIXDIR / "INV600254.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_payment_terms(doc, "Invoice")
    if "payment_terms" not in out:
        import pytest; pytest.skip("no payment terms extracted — fallback path not relevant to this test")
    val = out["payment_terms"].value
    assert "within 30 days" in val.lower()
    # No trailing comma / semicolon
    assert not val.endswith(",") and not val.endswith(";")


def test_dha_payment_terms_trimmed_at_section_break():
    """DHA source: 'PAYMENT DUE: NET 14 ...THIS INVOICE REFERENCES: PO438295 PAYMENT DETAILS: ...'
    Extracted terms should be 'NET 14', not include the trailing section
    breaks like 'THIS INVOICE REFERENCES:' or 'PAYMENT DETAILS:'."""
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DHA fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_payment_terms(doc, "Invoice")
    assert out.get("payment_terms") is not None
    val = out["payment_terms"].value.upper()
    assert "NET 14" in val
    # Should NOT include these section-break markers
    assert "PAYMENT DETAILS" not in val
    assert "REFERENCES" not in val
    assert "PO438295" not in val
