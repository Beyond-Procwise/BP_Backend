from src.services.structural_extractor.discovery.layout_fingerprint import layout_signature
from src.services.structural_extractor.parsing.model import ParsedDocument, Token, BBox


def _pdf_doc(tokens):
    return ParsedDocument(source_format="pdf", filename="", tokens=tokens, regions=[], tables=[],
                          pages_or_sheets=1, full_text="", raw_bytes=b"")


def test_same_layout_same_signature():
    toks1 = [Token(text="A", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    toks2 = [Token(text="B", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    assert layout_signature(_pdf_doc(toks1)) == layout_signature(_pdf_doc(toks2))


def test_different_layout_different_signature():
    toks1 = [Token(text="A", anchor=BBox(1, 10, 10, 20, 20), order=0)]
    toks2 = [Token(text="A", anchor=BBox(1, 500, 500, 510, 510), order=0)]
    assert layout_signature(_pdf_doc(toks1)) != layout_signature(_pdf_doc(toks2))
