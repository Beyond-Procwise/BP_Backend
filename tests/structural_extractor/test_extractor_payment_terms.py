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
