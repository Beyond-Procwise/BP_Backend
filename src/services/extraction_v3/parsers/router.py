"""Route a document to the right parser backend by file extension and scan-detection."""
from pathlib import Path
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from .scanned_classifier import is_scanned_pdf


def parse(path: Path | str) -> ParsedDocument:
    """Route a document to the appropriate parser backend.

    Routes based on file extension and (for PDFs) whether the document is scanned:
    - .pdf with selectable text → docling backend (native PDF)
    - .pdf with image-only content → paddleocr backend (scanned PDF)
    - .docx → docling backend
    - .png, .jpg, .jpeg → paddleocr backend

    Backends are imported lazily inside the branches so this module doesn't depend
    on their implementations being available yet.

    Args:
        path: Path to the document file.

    Returns:
        A ParsedDocument with extracted text and structural metadata.

    Raises:
        ValueError: If the file format is not supported.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        if is_scanned_pdf(p):
            from .paddleocr_backend import parse_with_paddleocr
            return parse_with_paddleocr(p, file_format="pdf-scanned")
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="pdf-native")

    if suffix == ".docx":
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="docx")

    if suffix in (".png", ".jpg", ".jpeg"):
        from .paddleocr_backend import parse_with_paddleocr
        return parse_with_paddleocr(p, file_format="image")

    raise ValueError(f"unsupported file format: {suffix}")
