"""PaddleOCR PP-Structure adapter: scanned PDF + image → ParsedDocument.

GPU-required (constraint C2 of the spec). On import the PP-Structure
pipeline is NOT loaded; first call to parse_with_paddleocr lazy-loads it.

paddleocr 3.5.0 / paddlepaddle-gpu 3.1.0 observed API
-------------------------------------------------------
``PPStructureV3().predict(img)`` returns a **list** of
``paddlex.inference.pipelines.layout_parsing.result_v2.LayoutParsingResultV2``
objects — one per image passed (so exactly one when we call with a single
numpy array).

Each result is a dict-like object with the following relevant keys:

  width, height          (int)  — pixel dimensions of the image
  parsing_res_list       (list of LayoutBlock) — ordered structural blocks
  overall_ocr_res        (dict-like OCRResult)  — fine-grained per-line OCR

LayoutBlock attributes (accessed via .to_dict()):
  label         str  — "text", "paragraph_title", "table", "figure", etc.
  order_label   str  — same taxonomy as label for non-table blocks
  bbox          list — [x0, y0, x1, y1] in pixel coords (top-left origin)
  content       str  — OCR'd text for text blocks; HTML string for tables

overall_ocr_res keys:
  rec_texts     list[str]          — one text per detected text-line
  rec_scores    list[float]        — confidence per text-line (0–1)
  rec_boxes     np.ndarray shape=(N,4) — [x0,y0,x1,y1] per text-line

Table structure is embedded in LayoutBlock.content as an HTML string
(<html><body><table>…</table></body></html>).  We parse this with the
stdlib html.parser to reconstruct rows×cells without an extra dependency.
"""

import html.parser
import logging
import threading
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

from src.services.extraction_v3.schemas.parsed_document import (
    BBox,
    Cell,
    Page,
    ParsedDocument,
    Region,
    Table,
    Token,
)

# ---------------------------------------------------------------------------
# Pipeline singleton — lazy GPU load, double-checked lock
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    """Return the lazy GPU-loaded PP-Structure pipeline (thread-safe singleton)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                import paddle  # C2: assert GPU is available before loading model
                if not paddle.device.get_device().startswith("gpu"):
                    raise RuntimeError(
                        f"PaddleOCR must run on GPU per spec C2; "
                        f"current device={paddle.device.get_device()}"
                    )
                from paddleocr import PPStructureV3  # deferred import — no GPU at collection time
                _pipeline = PPStructureV3()
    return _pipeline


# ---------------------------------------------------------------------------
# HTML table parser — extracts rows×cells from PP-Structure HTML output
# ---------------------------------------------------------------------------

class _TableHTMLParser(html.parser.HTMLParser):
    """Minimal SAX-style parser that extracts <td>/<th> text into a 2-D list."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell_text: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell_text = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._current_row is not None:
            self._current_row.append(self._current_cell_text or "")
            self._current_cell_text = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell_text is not None:
            self._current_cell_text += data


def _html_to_rows(html_content: str) -> list[list[str]]:
    """Parse HTML table string → list of rows, each row a list of cell texts.

    Returns an empty list on any parse error so a malformed table on one
    page does not abort the whole document.
    """
    try:
        parser = _TableHTMLParser()
        parser.feed(html_content)
        return parser.rows
    except Exception as e:
        logger.warning("_html_to_rows: failed to parse table HTML — skipping table. %s", e)
        return []


# ---------------------------------------------------------------------------
# PDF rasterisation helper
# ---------------------------------------------------------------------------

def _rasterize_pdf_pages(pdf_path: Path, dpi: int = 300) -> list[np.ndarray]:
    """Render every page of a PDF to an RGB numpy array."""
    from pdf2image import convert_from_path  # deferred — only needed for scanned PDFs
    pil_pages = convert_from_path(str(pdf_path), dpi=dpi)
    return [np.array(im) for im in pil_pages]


# ---------------------------------------------------------------------------
# Text-block label → Region.role mapping
# ---------------------------------------------------------------------------

_LABEL_TO_ROLE: dict[str, str] = {
    "figure": "logo",
    "logo": "logo",
    "stamp": "signature",
    "paragraph_title": "header",
    "abstract": "body",
    "header": "header",
    "footer": "footer",
}

_TABLE_LABELS = {"table"}
_FIGURE_LABELS = {"figure", "logo", "stamp"}


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_with_paddleocr(
    path: Path | str,
    file_format: Literal["pdf-scanned", "image"],
) -> ParsedDocument:
    """Convert a scanned PDF or raster image to a ParsedDocument.

    Uses GPU (NVIDIA A10G / PaddlePaddle-GPU 3.1.0 cu123).
    Rasterises PDFs at 300 DPI; images are used as-is.
    """
    p = Path(path)
    pipeline = _get_pipeline()

    # Collect per-page numpy images
    if file_format == "pdf-scanned":
        page_images = _rasterize_pdf_pages(p)
    else:
        from PIL import Image  # deferred
        _img = Image.open(p)
        # PaddleOCR normalizer expects exactly 3 channels (RGB).
        # Convert RGBA/grayscale images to RGB to avoid IndexError.
        if _img.mode != "RGB":
            _img = _img.convert("RGB")
        page_images = [np.array(_img)]

    pages: list[Page] = []
    # C1: full_text is built exclusively from rec_texts (per-line OCR) so that
    # every Token.text is guaranteed to be a substring of full_text.
    all_rec_texts: list[str] = []
    all_confidences: list[float] = []

    for pg_idx, img in enumerate(page_images):
        # PP-Structure predict returns a list with one item per input image
        # paddleocr 3.5.0 returns: list[LayoutParsingResultV2]
        result = pipeline.predict(img)
        page_result = result[0]  # single image → single result

        # --- Dimensions ---
        w = float(page_result["width"])
        h = float(page_result["height"])

        # --- Per-line OCR data for Token objects ---
        ocr_res = page_result["overall_ocr_res"]
        rec_texts: list[str] = ocr_res.get("rec_texts") or []
        rec_scores: list[float] = ocr_res.get("rec_scores") or []
        # rec_boxes: np.ndarray shape=(N,4) → [x0,y0,x1,y1]
        # Use explicit None-check because a non-empty ndarray is truthy-ambiguous with `or`.
        _raw_boxes = ocr_res.get("rec_boxes")
        rec_boxes: np.ndarray = np.asarray(_raw_boxes) if _raw_boxes is not None else np.empty((0, 4))

        page_tokens: list[Token] = []
        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            if not text:
                continue
            x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            page_tokens.append(Token(text=text, page=pg_idx, bbox=(x0, y0, x1, y1)))
            # C1: accumulate rec_texts in document order — this becomes full_text
            all_rec_texts.append(text)
            all_confidences.append(float(score))

        # --- Structural layout blocks for Regions and Tables ---
        parsing_res_list = page_result["parsing_res_list"]

        page_regions: list[Region] = []
        page_tables: list[Table] = []

        for block in parsing_res_list:
            # LayoutBlock.to_dict() returns plain Python dict with all fields
            bd = block.to_dict()
            label: str = bd.get("label", "text") or "text"
            raw_bbox = bd.get("bbox") or [0, 0, 0, 0]
            content: str = bd.get("content") or ""
            x0b, y0b, x1b, y1b = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
            block_bbox: BBox = (x0b, y0b, x1b, y1b)

            if label in _TABLE_LABELS:
                # Build Table from HTML-encoded content
                rows_text = _html_to_rows(content)
                rows: list[list[Cell]] = []
                for r_idx, row_cells_text in enumerate(rows_text):
                    row: list[Cell] = []
                    for c_idx, cell_text in enumerate(row_cells_text):
                        row.append(
                            Cell(
                                page=pg_idx,
                                bbox=block_bbox,  # table-level bbox (no per-cell bbox from parsing_res_list)
                                text=cell_text,
                                row_index=r_idx,
                                col_index=c_idx,
                            )
                        )
                    rows.append(row)
                page_tables.append(
                    Table(
                        page=pg_idx,
                        bbox=block_bbox,
                        rows=rows,
                        header_row_index=0 if rows else None,
                    )
                )
            elif label in _FIGURE_LABELS:
                role = _LABEL_TO_ROLE.get(label, "logo")
                page_regions.append(
                    Region(page=pg_idx, bbox=block_bbox, role=role, text="")
                )
            else:
                # text / paragraph_title / header / footer / abstract → Region
                role = _LABEL_TO_ROLE.get(label, "body")
                page_regions.append(
                    Region(page=pg_idx, bbox=block_bbox, role=role, text=content)
                )

        pages.append(
            Page(
                index=pg_idx,
                width=w,
                height=h,
                rotation=0,
                regions=page_regions,
                tables=page_tables,
                tokens=page_tokens,
            )
        )

    # I3: 0.0 (not 0.5) when no text detected so the router can fall back to Donut.
    overall_conf = (
        float(sum(all_confidences) / len(all_confidences)) if all_confidences else 0.0
    )

    # C1: full_text is the canonical join of rec_texts collected above.
    # Every Token.text is guaranteed to be a substring of this string.
    full_text = "\n".join(all_rec_texts)

    return ParsedDocument(
        source_path=str(p),
        file_format=file_format,
        pages=pages,
        full_text=full_text,
        parser_backend="paddleocr",
        parser_confidence=overall_conf,
    )
