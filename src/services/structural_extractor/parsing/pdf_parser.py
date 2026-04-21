from io import BytesIO
import logging
import fitz
import pdfplumber
from src.services.structural_extractor.parsing.model import (
    BBox, Token, Region, ParsedDocument, Table
)
from src.services.structural_extractor.exceptions import PDFParseError

log = logging.getLogger(__name__)


def _extract_tables(file_bytes: bytes) -> list[Table]:
    tables: list[Table] = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                for t in (page.extract_tables() or []):
                    if not t or not any(any(cell for cell in row) for row in t):
                        continue
                    rows: list[list[Region]] = []
                    for row_data in t:
                        row_regions = [
                            Region(
                                tokens=[Token(
                                    text=str(cell or ""),
                                    anchor=BBox(page_idx, 0.0, 0.0, 0.0, 0.0),
                                    order=0,
                                )] if (cell or "") else [],
                                kind="cell",
                            )
                            for cell in row_data
                        ]
                        rows.append(row_regions)
                    tables.append(Table(rows=rows, header_row_index=0 if rows else None))
    except Exception:
        log.debug("pdfplumber table extraction failed", exc_info=True)
    return tables


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"PyMuPDF failed to open {filename}: {exc}") from exc

    tokens: list[Token] = []
    regions: list[Region] = []
    full_text_parts: list[str] = []
    order = 0
    page_count = len(doc)

    for page_num, page in enumerate(doc, start=1):
        words = page.get_text("words")
        page_regions_by_block: dict[int, list[Token]] = {}
        for w in words:
            x0, y0, x1, y1, text, block, line, _ = w
            if not text.strip():
                continue
            tok = Token(
                text=text.strip(),
                anchor=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                block_no=block,
                line_no=line,
                order=order,
            )
            order += 1
            tokens.append(tok)
            page_regions_by_block.setdefault(block, []).append(tok)
        for _, block_tokens in page_regions_by_block.items():
            regions.append(Region(tokens=block_tokens, kind="block"))
        full_text_parts.append(page.get_text("text"))

    doc.close()

    return ParsedDocument(
        source_format="pdf",
        filename=filename,
        tokens=tokens,
        regions=regions,
        tables=_extract_tables(file_bytes),
        pages_or_sheets=page_count,
        full_text="\n".join(full_text_parts),
        raw_bytes=file_bytes,
    )
