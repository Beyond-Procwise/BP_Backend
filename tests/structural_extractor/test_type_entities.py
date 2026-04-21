from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.discovery.type_entities import find_candidates, Candidate
from src.services.structural_extractor.parsing.model import Token, BBox, ParsedDocument


def _tok(text, order=0):
    return Token(text=text, anchor=BBox(1, 0, 0, 0, 0), order=order)


def _doc(token_texts):
    toks = [_tok(t, i) for i, t in enumerate(token_texts)]
    return ParsedDocument(
        source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
        pages_or_sheets=1, full_text=" ".join(token_texts), raw_bytes=b"",
    )


def test_date_candidates():
    d = _doc(["Invoice", "Date:", "01/07/2022", "Random"])
    cands = find_candidates(d, FieldType.DATE)
    assert any(c.text == "01/07/2022" for c in cands)


def test_money_candidates():
    d = _doc(["Subtotal", "£8,333.00", "Tax", "20%", "Total", "£9,999.60"])
    cands = find_candidates(d, FieldType.MONEY)
    texts = {c.text for c in cands}
    assert "£8,333.00" in texts and "£9,999.60" in texts


def test_percent_candidate():
    d = _doc(["Tax", "(20%)"])
    cands = find_candidates(d, FieldType.PERCENT)
    assert any("20" in c.text for c in cands)


def test_currency_code_candidate():
    d = _doc(["Total", "9999.60", "GBP"])
    cands = find_candidates(d, FieldType.CURRENCY_CODE)
    assert any(c.text == "GBP" for c in cands)


def test_address_detected_from_uk_postcode():
    # Address block: 2+ lines with a UK postcode
    toks = [
        Token(text="10", anchor=BBox(1, 0, 10, 0, 0), line_no=0, order=0),
        Token(text="Redkiln", anchor=BBox(1, 0, 10, 0, 0), line_no=0, order=1),
        Token(text="Way", anchor=BBox(1, 0, 10, 0, 0), line_no=0, order=2),
        Token(text="Horsham", anchor=BBox(1, 0, 20, 0, 0), line_no=1, order=3),
        Token(text="RH13", anchor=BBox(1, 0, 30, 0, 0), line_no=2, order=4),
        Token(text="5QH", anchor=BBox(1, 0, 30, 0, 0), line_no=2, order=5),
    ]
    d = ParsedDocument(source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
                       pages_or_sheets=1, full_text=" ".join(t.text for t in toks), raw_bytes=b"")
    cands = find_candidates(d, FieldType.ADDRESS)
    assert len(cands) >= 1
