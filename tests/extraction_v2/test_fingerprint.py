"""Tests for layout fingerprinting."""
from __future__ import annotations

from src.services.extraction_v2.fingerprint import compute_fingerprint
from src.services.structural_extractor.parsing.model import (
    BBox, ParsedDocument, Region, Table, Token,
)


def _doc(filename: str, full_text: str, source_format: str = "pdf",
         tables: list[Table] | None = None) -> ParsedDocument:
    return ParsedDocument(
        source_format=source_format,
        filename=filename,
        tokens=[],
        regions=[],
        tables=tables or [],
        pages_or_sheets=1,
        full_text=full_text,
        raw_bytes=b"",
    )


class TestFingerprintBasics:
    def test_returns_hex_string(self):
        fp = compute_fingerprint(_doc("a.pdf", "Invoice\nTotal: 100"))
        assert isinstance(fp, str)
        assert all(c in "0123456789abcdef" for c in fp)
        assert len(fp) >= 16

    def test_same_layout_same_fingerprint(self):
        # Two docs with the same layout but different VALUES should
        # produce the same fingerprint — fingerprint captures structure.
        d1 = _doc("a.pdf", "Invoice No: A\nDate: 2024-01-01\nTotal: 100")
        d2 = _doc("b.pdf", "Invoice No: B\nDate: 2025-02-02\nTotal: 200")
        assert compute_fingerprint(d1) == compute_fingerprint(d2)

    def test_different_labels_different_fingerprint(self):
        d1 = _doc("a.pdf", "Invoice No: 1\nDate: 2024")
        d2 = _doc("a.pdf", "PO Number: 1\nOrdered: 2024")
        assert compute_fingerprint(d1) != compute_fingerprint(d2)

    def test_format_in_fingerprint(self):
        # Same labels, different format → different fingerprint
        d1 = _doc("a.pdf", "Invoice No: 1", source_format="pdf")
        d2 = _doc("a.xlsx", "Invoice No: 1", source_format="xlsx")
        assert compute_fingerprint(d1) != compute_fingerprint(d2)

    def test_table_shape_affects_fingerprint(self):
        t1 = Table(rows=[[Region([], "cell")] * 3] * 2, header_row_index=0)
        t2 = Table(rows=[[Region([], "cell")] * 5] * 2, header_row_index=0)
        d1 = _doc("a.pdf", "Invoice", tables=[t1])
        d2 = _doc("a.pdf", "Invoice", tables=[t2])
        assert compute_fingerprint(d1) != compute_fingerprint(d2)


class TestFingerprintStability:
    def test_deterministic(self):
        d = _doc("a.pdf", "Invoice No: 1\nTotal: 100")
        fp1 = compute_fingerprint(d)
        fp2 = compute_fingerprint(d)
        assert fp1 == fp2

    def test_value_changes_dont_affect_fingerprint(self):
        # Numbers and dates are erased in label set
        d1 = _doc("a.pdf", "Invoice No: 12345\nDate: 2024-01-01")
        d2 = _doc("a.pdf", "Invoice No: 99999\nDate: 2025-12-31")
        assert compute_fingerprint(d1) == compute_fingerprint(d2)

    def test_filename_does_not_affect_fingerprint(self):
        # Same layout, different filename → same fp
        d1 = _doc("AQUARIUS-INV-001.pdf", "Invoice No: 1\nTotal: 100")
        d2 = _doc("AQUARIUS-INV-002.pdf", "Invoice No: 2\nTotal: 200")
        assert compute_fingerprint(d1) == compute_fingerprint(d2)

    def test_empty_doc_returns_consistent_fingerprint(self):
        d1 = _doc("a.pdf", "")
        d2 = _doc("b.pdf", "")
        assert compute_fingerprint(d1) == compute_fingerprint(d2)
