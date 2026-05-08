import pytest
from pydantic import ValidationError
from src.services.extraction_v3.schemas.parsed_document import (
    ParsedDocument, Page, Region, Token, Cell, Table
)

def test_parsed_document_minimal():
    doc = ParsedDocument(
        source_path="/tmp/x.pdf",
        file_format="pdf-native",
        pages=[Page(index=0, width=612, height=792, rotation=0,
                    regions=[], tables=[], tokens=[])],
        full_text="",
        parser_backend="docling",
        parser_confidence=1.0,
    )
    assert doc.pages[0].index == 0

def test_parsed_document_rejects_invalid_rotation():
    with pytest.raises(ValidationError):
        Page(index=0, width=1, height=1, rotation=45,
             regions=[], tables=[], tokens=[])

def test_token_bbox_is_4_floats():
    t = Token(text="x", page=0, bbox=(0.0, 0.0, 1.0, 1.0), font_size=12.0, is_bold=False)
    assert t.bbox == (0.0, 0.0, 1.0, 1.0)
