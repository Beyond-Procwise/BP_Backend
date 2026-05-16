"""L0 — unified parser entry. Delegates to extraction_v3.parsers.router.parse.

This is intentionally thin: the existing router already handles per-format
backend selection (docling / paddleocr / donut) and scanned classification.
The wrapper exists so the rest of the renovation imports a stable
`extraction.parser.parse` symbol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from src.services.extraction_v3.parsers.router import parse as _route_parse
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument


def parse(file_path: Union[str, Path]) -> ParsedDocument:
    """Return a ParsedDocument for the given file path.

    Supported formats: PDF (native + scanned), DOCX, PNG, JPG, JPEG.
    Raises ValueError for unsupported formats.
    """
    return _route_parse(Path(file_path))
