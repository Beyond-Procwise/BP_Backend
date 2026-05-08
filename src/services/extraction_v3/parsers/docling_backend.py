"""Docling adapter: parse a native PDF or DOCX into our ParsedDocument schema.

Docling is IBM's fully local document converter.  It uses HuggingFace models
cached locally on first run — there is no cloud call at parse time.

Coordinate notes (Docling 2.x):
- Text item prov[].bbox uses BOTTOMLEFT origin  → we normalise to TOPLEFT
  via bbox.to_top_left_origin(page_height).
- TableCell.bbox uses TOPLEFT origin already.
- For DOCX files Docling has no page coordinate system: prov lists are empty
  and doc.pages == {}.  All DOCX tokens receive placeholder bbox (0,0,0,0)
  and the entire document is placed on a single synthetic Page(index=0).
"""

from pathlib import Path
from typing import Literal

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import CoordOrigin

from src.services.extraction_v3.schemas.parsed_document import (
    BBox,
    Cell,
    Page,
    ParsedDocument,
    Region,
    Table,
    Token,
)

# Module-level singleton — HuggingFace models load once per process / pytest session.
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def _to_topleft_bbox(bbox, page_height: float) -> BBox:
    """Return (l, t, r, b) in top-left origin coordinates."""
    if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
        tl = bbox.to_top_left_origin(page_height)
        return (tl.l, tl.t, tl.r, tl.b)
    # Already TOPLEFT
    return (bbox.l, bbox.t, bbox.r, bbox.b)


def parse_with_docling(
    path: Path | str,
    file_format: Literal["pdf-native", "docx"],
) -> ParsedDocument:
    """Convert *path* with Docling and return a ``ParsedDocument``.

    For PDFs every text span and table cell is given its real bounding-box in
    top-left coordinates.  For DOCX files Docling does not expose a coordinate
    system, so all tokens carry a placeholder bbox of (0, 0, 0, 0) and the
    whole document is wrapped in a single synthetic page.
    """
    p = Path(path)
    result = _get_converter().convert(p)
    docling_doc = result.document

    # ------------------------------------------------------------------ #
    # Full reading-order text (markdown is the most faithful export)       #
    # ------------------------------------------------------------------ #
    full_text: str = docling_doc.export_to_markdown() or ""

    # ------------------------------------------------------------------ #
    # Page dimensions (PDF only — DOCX pages dict is empty)               #
    # ------------------------------------------------------------------ #
    page_dims: dict[int, tuple[float, float]] = {}
    for page_no_1, pg in docling_doc.pages.items():
        pn = page_no_1 - 1  # Docling is 1-indexed; we use 0-indexed
        sz = getattr(pg, "size", None)
        w = sz.width if sz else 612.0
        h = sz.height if sz else 792.0
        page_dims[pn] = (w, h)

    # ------------------------------------------------------------------ #
    # Tokens — built from docling_doc.texts                               #
    # Each TextItem carries a list of ProvenanceItem with bbox+page_no.   #
    # DOCX items have empty prov; those are collected onto page 0 with    #
    # placeholder bbox.                                                    #
    # ------------------------------------------------------------------ #
    page_tokens: dict[int, list[Token]] = {}

    for text_item in docling_doc.texts:
        text = getattr(text_item, "text", "") or ""
        if not text:
            continue

        provs = getattr(text_item, "prov", []) or []
        if provs:
            for prov in provs:
                pn = prov.page_no - 1
                w, h = page_dims.get(pn, (612.0, 792.0))
                bbox = _to_topleft_bbox(prov.bbox, h)
                token = Token(
                    text=text,
                    page=pn,
                    bbox=bbox,
                    font_size=None,
                    is_bold=False,
                )
                page_tokens.setdefault(pn, []).append(token)
        else:
            # DOCX path: no coordinates available
            token = Token(
                text=text,
                page=0,
                bbox=(0.0, 0.0, 0.0, 0.0),
                font_size=None,
                is_bold=False,
            )
            page_tokens.setdefault(0, []).append(token)

    # ------------------------------------------------------------------ #
    # Tables — built from docling_doc.tables                              #
    # ------------------------------------------------------------------ #
    page_tables: dict[int, list[Table]] = {}

    for table_item in docling_doc.tables:
        provs = getattr(table_item, "prov", []) or []
        pn = provs[0].page_no - 1 if provs else 0
        w, h = page_dims.get(pn, (612.0, 792.0))

        # Table-level bounding box (fallback for cells without individual bboxes)
        if provs:
            tbl_bbox = _to_topleft_bbox(provs[0].bbox, h)
        else:
            tbl_bbox = (0.0, 0.0, 0.0, 0.0)

        rows: list[list[Cell]] = []
        tdata = getattr(table_item, "data", None)
        if tdata and hasattr(tdata, "grid"):
            for r_idx, row in enumerate(tdata.grid):
                row_cells: list[Cell] = []
                for c_idx, cell_item in enumerate(row):
                    cell_text = getattr(cell_item, "text", "") or ""
                    cell_raw_bbox = getattr(cell_item, "bbox", None)
                    if cell_raw_bbox is not None:
                        # TableCell bboxes are already TOPLEFT
                        cell_bbox: BBox = (
                            cell_raw_bbox.l,
                            cell_raw_bbox.t,
                            cell_raw_bbox.r,
                            cell_raw_bbox.b,
                        )
                    else:
                        cell_bbox = tbl_bbox  # fall back to table bbox

                    # Docling uses offset indices for spans
                    row_span = max(
                        1,
                        (getattr(cell_item, "end_row_offset_idx", r_idx + 1) or r_idx + 1)
                        - (getattr(cell_item, "start_row_offset_idx", r_idx) or r_idx),
                    )
                    col_span = max(
                        1,
                        (getattr(cell_item, "end_col_offset_idx", c_idx + 1) or c_idx + 1)
                        - (getattr(cell_item, "start_col_offset_idx", c_idx) or c_idx),
                    )

                    row_cells.append(
                        Cell(
                            page=pn,
                            bbox=cell_bbox,
                            text=cell_text,
                            row_index=r_idx,
                            col_index=c_idx,
                            row_span=row_span,
                            col_span=col_span,
                        )
                    )
                rows.append(row_cells)

        tbl = Table(
            page=pn,
            bbox=tbl_bbox,
            rows=rows,
            header_row_index=0 if rows else None,
        )
        page_tables.setdefault(pn, []).append(tbl)

    # ------------------------------------------------------------------ #
    # Compose Page list                                                    #
    # ------------------------------------------------------------------ #
    all_page_nos = sorted(
        set(page_tokens) | set(page_tables) | set(page_dims) | {0}
    )
    pages: list[Page] = []
    for pn in all_page_nos:
        w, h = page_dims.get(pn, (612.0, 792.0))
        pages.append(
            Page(
                index=pn,
                width=w,
                height=h,
                rotation=0,
                regions=[],  # Docling doesn't classify header/footer/body regions
                tables=page_tables.get(pn, []),
                tokens=page_tokens.get(pn, []),
            )
        )

    return ParsedDocument(
        source_path=str(p),
        file_format=file_format,
        pages=pages,
        full_text=full_text,
        parser_backend="docling",
        parser_confidence=1.0,
    )
