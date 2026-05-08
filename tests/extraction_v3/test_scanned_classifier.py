"""Tests for the scanned PDF classifier."""
from pathlib import Path
from src.services.extraction_v3.parsers.scanned_classifier import is_scanned_pdf
import pytest

FX = Path(__file__).parent / "fixtures/invoices"


def test_native_pdf_not_scanned():
    """A PDF with selectable text should not be classified as scanned."""
    assert is_scanned_pdf(FX / "INV-001-clean.pdf") is False


def test_scanned_pdf_detected():
    """A PDF with image-only content should be classified as scanned."""
    assert is_scanned_pdf(FX / "INV-005-scanned.pdf") is True


def test_unsupported_extension_raises():
    """Router must raise ValueError on file types it doesn't support."""
    from src.services.extraction_v3.parsers.router import parse
    with pytest.raises(ValueError, match="unsupported file format"):
        parse(Path("/tmp/some_file.txt"))


def test_empty_pdf_classified_as_scanned(tmp_path):
    """A PDF with zero pages should be treated as scanned (defensive default)."""
    # Create a minimal valid PDF with zero pages
    pdf_path = tmp_path / "empty.pdf"
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [] /Count 0 >>
endobj
xref
0 3
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
trailer
<< /Size 3 /Root 1 0 R >>
startxref
117
%%EOF
"""
    pdf_path.write_bytes(minimal_pdf)
    assert is_scanned_pdf(pdf_path) is True
