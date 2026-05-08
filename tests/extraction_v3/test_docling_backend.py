"""Tests for the Docling parser backend (Task 5 of extraction-redesign plan).

The Docling converter is module-level cached — HuggingFace models load once
per pytest session, subsequent parses take ~1s.
"""

from pathlib import Path

import pytest

from src.services.extraction_v3.parsers.docling_backend import parse_with_docling

FX = Path(__file__).parent / "fixtures/invoices"


def test_docling_parses_native_pdf():
    doc = parse_with_docling(FX / "INV-001-clean.pdf", file_format="pdf-native")
    assert doc.parser_backend == "docling"
    assert doc.file_format == "pdf-native"
    assert len(doc.pages) >= 1
    assert doc.full_text  # non-empty
    # At least some tokens should carry real bounding boxes from the PDF
    assert any(t.bbox != (0.0, 0.0, 0.0, 0.0) for t in doc.pages[0].tokens), (
        "no tokens with real bbox on page 0"
    )


def test_docling_parses_docx():
    doc = parse_with_docling(FX / "INV-006-docx.docx", file_format="docx")
    assert doc.parser_backend == "docling"
    assert doc.file_format == "docx"
    assert doc.full_text  # non-empty
    # DOCX has no fixed coordinate system in Docling; tokens carry (0,0,0,0).
    # We still expect at least one synthetic page and at least one token.
    assert len(doc.pages) >= 1
    assert any(len(pg.tokens) > 0 for pg in doc.pages), "no tokens extracted from DOCX"
