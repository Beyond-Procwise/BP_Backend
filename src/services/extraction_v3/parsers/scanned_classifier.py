"""Detect whether a PDF is image-only (scanned) vs has selectable text."""
from pathlib import Path
import pdfplumber


def is_scanned_pdf(path: Path | str) -> bool:
    """Return True if the PDF has effectively no extractable text (≤ 5 chars/page average).

    A scanned PDF is one where the content is stored as images rather than selectable text.
    This classifier measures the average characters per page and considers the PDF scanned
    if that average is 5 or fewer characters. Handles edge cases like empty PDFs (no pages)
    which are considered scanned.

    Args:
        path: Path to the PDF file.

    Returns:
        True if the PDF is scanned (image-only); False if it has extractable text.
    """
    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            return True
        total_chars = sum(len((p.extract_text() or "")) for p in pdf.pages)
        return (total_chars / len(pdf.pages)) <= 5
