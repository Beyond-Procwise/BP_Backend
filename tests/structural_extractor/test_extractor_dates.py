from pathlib import Path
from src.services.structural_extractor.extractors.dates import extract_dates, detect_locale
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.parsing.model import Token, BBox, ParsedDocument

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_detect_locale_uk_from_postcode():
    toks = [
        Token(text="RH13", anchor=BBox(1, 0, 0, 0, 0), order=0),
        Token(text="5QH", anchor=BBox(1, 0, 0, 0, 0), order=1),
    ]
    d = ParsedDocument(source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
                       pages_or_sheets=1, full_text="RH13 5QH", raw_bytes=b"")
    assert detect_locale(d) == "dmy"


def test_extract_invoice_date_from_newport():
    if not FIX.exists():
        import pytest; pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    out = extract_dates(doc, doc_type="Invoice")
    assert "invoice_date" in out
    assert out["invoice_date"].value == "2019-08-22"
    assert out["invoice_date"].provenance == "extracted"
