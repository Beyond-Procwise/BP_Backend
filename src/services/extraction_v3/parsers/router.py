"""Route a document to the right parser backend by file extension and scan-detection.

Fallback rule: when PaddleOCR's parser_confidence falls below
PADDLE_LOW_CONF_THRESHOLD (0.6) for a scanned PDF or image, the router
retries with the Donut backend. The higher-confidence ParsedDocument wins;
a low-conf Donut result will NOT replace a higher-conf PaddleOCR result.
"""
from pathlib import Path
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from .scanned_classifier import is_scanned_pdf

PADDLE_LOW_CONF_THRESHOLD = 0.6


def parse(path: Path | str) -> ParsedDocument:
    """Route a document to the appropriate parser backend.

    Routes based on file extension and (for PDFs) whether the document is scanned:
    - .pdf with selectable text → docling backend (native PDF)
    - .pdf with image-only content → paddleocr backend (scanned PDF), with
      Donut fallback when paddle confidence < PADDLE_LOW_CONF_THRESHOLD
    - .docx → docling backend
    - .png, .jpg, .jpeg → paddleocr backend, with Donut fallback when
      paddle confidence < PADDLE_LOW_CONF_THRESHOLD

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
            doc = parse_with_paddleocr(p, file_format="pdf-scanned")
            if doc.parser_confidence < PADDLE_LOW_CONF_THRESHOLD:
                from .donut_backend import parse_with_donut
                fallback = parse_with_donut(p, file_format="pdf-scanned")
                # Only replace if Donut is more confident than PaddleOCR
                if fallback.parser_confidence > doc.parser_confidence:
                    return fallback
            return doc
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="pdf-native")

    if suffix == ".docx":
        from .docling_backend import parse_with_docling
        return parse_with_docling(p, file_format="docx")

    if suffix in (".png", ".jpg", ".jpeg"):
        from .paddleocr_backend import parse_with_paddleocr
        doc = parse_with_paddleocr(p, file_format="image")
        if doc.parser_confidence < PADDLE_LOW_CONF_THRESHOLD:
            from .donut_backend import parse_with_donut
            fallback = parse_with_donut(p, file_format="image")
            # Only replace if Donut is more confident than PaddleOCR
            if fallback.parser_confidence > doc.parser_confidence:
                return fallback
        return doc

    raise ValueError(f"unsupported file format: {suffix}")
