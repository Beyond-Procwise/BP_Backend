"""Tests for the scanned PDF classifier."""
from pathlib import Path
from src.services.extraction_v3.parsers.scanned_classifier import is_scanned_pdf

FX = Path(__file__).parent / "fixtures/invoices"


def test_native_pdf_not_scanned():
    """A PDF with selectable text should not be classified as scanned."""
    assert is_scanned_pdf(FX / "INV-001-clean.pdf") is False


def test_scanned_pdf_detected():
    """A PDF with image-only content should be classified as scanned."""
    assert is_scanned_pdf(FX / "INV-005-scanned.pdf") is True
