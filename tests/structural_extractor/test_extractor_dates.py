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


FIXDIR = Path(__file__).parent / "fixtures/docs"


def test_dha_invoice_date_bare_date_label():
    """DHA: 'DATE: 01/07/2022' — label is bare 'DATE:', not 'Invoice Date'.
    Must still extract as invoice_date = 2022-07-01 (UK dmy)."""
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DHA fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_dates(doc, doc_type="Invoice")
    assert out.get("invoice_date") is not None
    assert out["invoice_date"].value == "2022-07-01"


def test_eleanor_invoice_date_three_token_window():
    """ELEANOR: '1 April 2020' — must be parsed via the multi-token DATE
    candidate window and attributed to invoice_date."""
    fixture = FIXDIR / "ELEANOR-2025-290.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("ELEANOR fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_dates(doc, doc_type="Invoice")
    assert out.get("invoice_date") is not None
    assert out["invoice_date"].value == "2020-04-01"


def test_duncan_order_date_po_date_label():
    """DUNCAN: 'PO Date: 3 Jan 2024' — order_date should accept label 'PO Date'."""
    fixture = FIXDIR / "DUNCAN_PO526800.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DUNCAN fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_dates(doc, doc_type="Purchase_Order")
    assert out.get("order_date") is not None
    assert out["order_date"].value == "2024-01-03"


def test_no_today_fallback_when_text_has_no_year():
    """A date candidate must not silently default to today's date when
    dateutil can't parse the text. INV600254 currently works, but DHA
    was returning 2026-04-14 (today) for a candidate that didn't contain
    a valid year."""
    from datetime import date
    fixture = FIXDIR / "DHA-2025-143.pdf"
    if not fixture.exists():
        import pytest; pytest.skip("DHA fixture missing")
    doc = parse_pdf(fixture.read_bytes(), fixture.name)
    out = extract_dates(doc, doc_type="Invoice")
    today = date.today().isoformat()
    for field, ev in out.items():
        assert ev.value != today, f"{field} defaulted to today's date!"
