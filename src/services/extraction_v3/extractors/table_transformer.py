"""Table Transformer (microsoft/table-transformer-structure-recognition-v1.1-all)
extractor for line-item detection.

Plan-1 implementation: rasterize each page, run the detection model, group
cells into rows by row-bbox intersection, map columns to schema's line-item
fields by header-row label matching (using the same canonical-label
proximity used in layoutlmv3). Cell text is filled by intersecting cell
bboxes with the parsed Token list (same coordinate system after Docling's
to_top_left_origin normalization).

Substring guarantee: cell text is built by joining the texts of Tokens whose
bboxes intersect each cell. Tokens come from the parser, so cell text — and
therefore Candidate.evidence_text — is a substring of parsed.full_text by
construction.

Compatibility note: transformers >= 5.x has a strict HF Hub dataclass
validator that rejects the `dilation: null` in the model's config.json as
a boolean field. We work around this by loading the config dict, coercing
`dilation` to False, then constructing the config object manually before
loading the safetensors weights. The AutoImageProcessor for this model also
has a size-format mismatch (longest_edge key only); we use DetrImageProcessor
directly with an explicit {'height': 800, 'width': 800} size.

Coordinate system: Docling emits Token bboxes in PDF point space
(612 × 792 for letter pages). The rasterized image at DPI has a pixel
dimension of (page_width_pts * DPI/72) × (page_height_pts * DPI/72).
Model detections are in pixel space; we convert them back to point space
before token intersection so that the coordinate systems are aligned.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument, Token, Page
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"
DETECTION_THRESHOLD = 0.7
_RASTERIZE_DPI = 200

# id2label for this model version (also available via model.config.id2label)
_LABEL_TABLE_ROW = "table row"
_LABEL_TABLE_COL = "table column"
_LABEL_TABLE_COL_HEADER = "table column header"
_LABEL_TABLE_PROJ_ROW_HEADER = "table projected row header"

# Minimum fuzzy-match score (0–100) to assign a column to a schema field
_COL_MATCH_MIN_SCORE: int = 70

# ---------------------------------------------------------------------------
# Lazy singleton — model loads once per process
# ---------------------------------------------------------------------------
_proc = None
_model = None
_model_path: str | None = None
_lock = threading.Lock()


def _find_model_snapshot() -> str:
    """Return the local snapshot directory for the Table Transformer model.

    Walks the HF Hub cache so we can load the config.json and safetensors
    weights directly, bypassing the transformers high-level API that fails
    with the dilation:null bug.
    """
    import os
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"
    slug = MODEL_NAME.replace("/", "--")
    model_dir = hub_dir / f"models--{slug}"
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        dirs = sorted(snapshots.iterdir())
        if dirs:
            return str(dirs[-1])  # latest snapshot
    raise FileNotFoundError(
        f"Table Transformer model not cached at {model_dir}. "
        f"Run: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}')\""
    )


def _load_model_and_proc():
    """Load TableTransformerForObjectDetection + DetrImageProcessor.

    Works around two transformers-5.x bugs:
      1. `dilation: null` in config.json fails the strict bool validator —
         we coerce it to False before constructing TableTransformerConfig.
      2. `AutoImageProcessor` fails with `longest_edge`-only size dict —
         we use `DetrImageProcessor` directly with an explicit H×W size.
    """
    from transformers import (
        DetrImageProcessor,
        TableTransformerConfig,
        TableTransformerForObjectDetection,
    )
    from safetensors.torch import load_file

    snap = _find_model_snapshot()

    # --- config with dilation fix ---
    with open(Path(snap) / "config.json") as fh:
        cfg_dict = json.load(fh)
    if cfg_dict.get("dilation") is None:
        cfg_dict["dilation"] = False  # coerce None → False for strict bool validator

    config = TableTransformerConfig(**cfg_dict)
    model = TableTransformerForObjectDetection(config)

    # --- weights ---
    weights_path = Path(snap) / "model.safetensors"
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model = model.to("cuda").eval()

    # --- processor: explicit size avoids longest_edge mismatch ---
    proc = DetrImageProcessor.from_pretrained(snap, size={"height": 800, "width": 800})

    return proc, model


def _get_model():
    """Return (proc, model) singleton; loads on first call (thread-safe)."""
    global _proc, _model
    if _proc is not None and _model is not None:
        return _proc, _model
    with _lock:
        if _proc is None or _model is None:
            assert torch.cuda.is_available(), (
                "Table Transformer requires GPU per extraction spec C2. "
                "Set CUDA_VISIBLE_DEVICES or run on a GPU host."
            )
            _proc, _model = _load_model_and_proc()
    return _proc, _model


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _rasterize_page(parsed: ParsedDocument, page_index: int, dpi: int = _RASTERIZE_DPI) -> Image.Image:
    """Rasterize a single page from the source document.

    Supports PDF (via pdf2image / poppler) and standalone image files.
    DOCX files have no rasterization path — the caller skips them.
    """
    p = Path(parsed.source_path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from pdf2image import convert_from_path
        images = convert_from_path(
            str(p), dpi=dpi, first_page=page_index + 1, last_page=page_index + 1
        )
        return images[0]
    if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        return Image.open(p).convert("RGB")
    raise ValueError(
        f"Cannot rasterize {p.suffix!r} pages for Table Transformer. "
        "Only PDF and image files are supported."
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_table_structure(img: Image.Image) -> dict:
    """Run Table Transformer inference on *img*, return detections in PIXEL coords.

    Returns a dict with keys:
      scores  — numpy (N,) float32
      labels  — numpy (N,) int
      boxes   — numpy (N, 4) float32, (x0, y0, x1, y1) in pixels
      id2label — dict[int, str]
    """
    proc, model = _get_model()
    inputs = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to("cuda")  # (H, W)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=DETECTION_THRESHOLD
    )[0]
    return {
        "scores": results["scores"].cpu().numpy(),
        "labels": results["labels"].cpu().numpy(),
        "boxes": results["boxes"].cpu().numpy(),
        "id2label": model.config.id2label,
    }


def _pixel_to_points(bbox_px: tuple, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    """Convert a pixel-space bbox to PDF point space.

    scale_x = page_width_pts / img_width_px
    scale_y = page_height_pts / img_height_px
    """
    x0, y0, x1, y1 = bbox_px
    return (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)


def _group_into_rows(
    detections: dict,
    scale_x: float,
    scale_y: float,
) -> list[dict]:
    """Parse flat detections into a sorted list of row dicts.

    Each dict has:
      bbox     — (x0, y0, x1, y1) in PDF point space
      columns  — list of column bboxes in PDF point space, sorted left→right

    Table Transformer label IDs (v1.1-all):
      0: table
      1: table column
      2: table row
      3: table column header
      4: table projected row header
      5: table spanning cell
    """
    id2label = detections["id2label"]
    rows_by_y: list[tuple[float, tuple]] = []
    cols_by_x: list[tuple[float, tuple]] = []

    for box_px, label_id, score in zip(
        detections["boxes"], detections["labels"], detections["scores"]
    ):
        label = id2label[int(label_id)]
        bbox_pts = _pixel_to_points(tuple(map(float, box_px)), scale_x, scale_y)

        if label in (_LABEL_TABLE_ROW, _LABEL_TABLE_PROJ_ROW_HEADER):
            rows_by_y.append((bbox_pts[1], bbox_pts))  # sort key = y0
        elif label in (_LABEL_TABLE_COL, _LABEL_TABLE_COL_HEADER):
            cols_by_x.append((bbox_pts[0], bbox_pts))  # sort key = x0

    rows_by_y.sort(key=lambda t: t[0])
    cols_by_x.sort(key=lambda t: t[0])
    col_bboxes = [cb for _, cb in cols_by_x]

    return [{"bbox": rb, "columns": col_bboxes} for _, rb in rows_by_y]


def _intersect(
    b1: tuple[float, float, float, float],
    b2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the intersection of two (x0,y0,x1,y1) bboxes."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    return (x0, y0, x1, y1)


def _bbox_has_area(bbox: tuple[float, float, float, float]) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _tokens_in_bbox(
    tokens: list[Token],
    bbox: tuple[float, float, float, float],
    pad: float = 2.0,
) -> list[Token]:
    """Return tokens whose centroid falls inside *bbox* (with optional padding)."""
    x0, y0, x1, y1 = bbox
    out: list[Token] = []
    for t in tokens:
        tx0, ty0, tx1, ty1 = t.bbox
        cx = (tx0 + tx1) / 2.0
        cy = (ty0 + ty1) / 2.0
        if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

@register_extractor("table_transformer")
class TableTransformerExtractor(Extractor):
    """Line-item candidate generator backed by Table Transformer.

    Algorithm:
      1. For each page in parsed.pages, rasterize to a PIL Image.
      2. Run Table Transformer → row/column bboxes in pixel space.
      3. Convert detections to PDF point space (same CRS as Token.bbox).
      4. Treat the first row as the header row; match each column to a
         line-item schema field by fuzzy-matching the header cell text
         against field.canonical_labels.
      5. For each subsequent (data) row × matched column, intersect the
         cell bbox with page.tokens, join token texts → cell_text.
      6. Verify cell_text is a substring of parsed.full_text (substring
         guarantee), then emit a Candidate.
    """

    # Keywords that identify non-data rows (header-area / footer / address rows).
    # If ANY cell in a row contains one of these keywords (case-insensitive),
    # the entire row is skipped so label-shaped text never lands in data cells.
    _SKIP_ROW_KEYWORDS = frozenset({
        "bill to", "ship to", "deliver to", "sold to", "attention",
        "subtotal", "sub total", "sub-total",
        "total amount", "grand total", "invoice total", "amount due",
        "total incl", "total excl",
        "tax", "vat", "gst", "hst",
        "discount", "freight", "shipping",
        "thank you", "payment", "please",
        "description", "item description", "product", "service",
        "qty", "quantity", "unit price", "rate", "amount", "extended",
        "bill to:", "ship to:", "deliver to:",
    })

    def produce_candidates(
        self, parsed: ParsedDocument, schema: DocSchema
    ) -> list[Candidate]:
        if schema.line_items is None:
            return []
        line_fields = schema.line_items.fields
        # Build a field_type lookup for type-validation of cell values
        field_type_map: dict[str, str] = {f.name: f.type for f in line_fields}
        candidates: list[Candidate] = []

        for page in parsed.pages:
            try:
                img = _rasterize_page(parsed, page.index)
            except (ValueError, FileNotFoundError):
                # DOCX pages or unsupported formats — skip gracefully
                continue

            detections = _detect_table_structure(img)

            # Scale factors: convert pixel → PDF point space
            img_w_px, img_h_px = img.size
            scale_x = page.width / img_w_px if img_w_px > 0 else 1.0
            scale_y = page.height / img_h_px if img_h_px > 0 else 1.0

            rows = _group_into_rows(detections, scale_x, scale_y)
            if len(rows) < 2:
                # Need at least one header row + one data row
                continue

            header_row, *data_rows = rows

            # Map columns → schema fields by header text proximity
            header_tokens_per_col = [
                _tokens_in_bbox(page.tokens, _intersect(header_row["bbox"], col_bbox))
                for col_bbox in header_row["columns"]
            ]
            col_to_field = self._match_columns_to_fields(header_tokens_per_col, line_fields)

            if not col_to_field:
                continue  # no columns matched any schema field

            # Emit candidates for data rows
            for row_idx, row in enumerate(data_rows):
                # Collect all cell texts for this row to check for label keywords
                row_cell_texts: list[str] = []
                for col_idx, col_bbox in enumerate(row["columns"]):
                    cell_bbox = _intersect(row["bbox"], col_bbox)
                    if not _bbox_has_area(cell_bbox):
                        row_cell_texts.append("")
                        continue
                    cell_tokens = _tokens_in_bbox(page.tokens, cell_bbox)
                    cell_tokens.sort(key=lambda t: t.bbox[0])
                    row_cell_texts.append(" ".join(t.text for t in cell_tokens).strip())

                # Skip the entire row if any cell contains a non-data keyword
                combined_row_text = " ".join(row_cell_texts).lower()
                if self._row_is_label_like(combined_row_text):
                    continue

                for col_idx, col_bbox in enumerate(row["columns"]):
                    field = col_to_field.get(col_idx)
                    if field is None:
                        continue

                    cell_bbox = _intersect(row["bbox"], col_bbox)
                    if not _bbox_has_area(cell_bbox):
                        continue

                    cell_tokens = _tokens_in_bbox(page.tokens, cell_bbox)
                    if not cell_tokens:
                        continue

                    # Sort tokens left→right within cell for natural reading order
                    cell_tokens.sort(key=lambda t: t.bbox[0])
                    cell_text = " ".join(t.text for t in cell_tokens).strip()

                    if not cell_text:
                        continue

                    # Substring guarantee: cell_text must appear literally in full_text
                    if cell_text not in parsed.full_text:
                        # Fall back: try each token individually
                        for tok in cell_tokens:
                            if tok.text.strip() and tok.text.strip() in parsed.full_text:
                                cell_text = tok.text.strip()
                                cell_bbox = tok.bbox
                                break
                        else:
                            continue  # no grounded text found → skip

                    # Type-validation: for typed fields (money/decimal/date),
                    # reject cell values that don't parse to the expected type.
                    field_type = field_type_map.get(field.name, "string")
                    if not self._cell_passes_type_check(cell_text, field_type):
                        continue

                    candidates.append(
                        Candidate(
                            field=f"line_items[{row_idx}].{field.name}",
                            value=cell_text,
                            page=page.index,
                            bbox=cell_bbox,
                            evidence_text=cell_text,
                            model="table_transformer",
                            confidence=0.85,
                        )
                    )

        return candidates

    def _row_is_label_like(self, combined_row_text_lower: str) -> bool:
        """Return True if this row looks like a header/footer/address row."""
        for kw in self._SKIP_ROW_KEYWORDS:
            if kw in combined_row_text_lower:
                return True
        return False

    def _cell_passes_type_check(self, cell_text: str, field_type: str) -> bool:
        """Return True if cell_text is plausible for field_type."""
        if field_type in ("money", "decimal"):
            from src.services.extraction_v2.parsers.amounts import parse_amount
            return parse_amount(cell_text) is not None
        if field_type == "iso_date":
            from src.services.extraction_v2.parsers.dates import parse_date
            return parse_date(cell_text) is not None
        # string fields: just need non-empty, non-pure-label text
        return bool(cell_text.strip())

    # ---------------------------------------------------------------------- #
    # Column → field matching                                                  #
    # ---------------------------------------------------------------------- #

    def _match_columns_to_fields(
        self,
        header_tokens_per_col: list[list[Token]],
        line_fields: list[FieldSpec],
    ) -> dict[int, FieldSpec]:
        """Match each header column to a line-item FieldSpec by fuzzy label comparison.

        Uses the same rapidfuzz partial_ratio approach as LayoutLMv3's
        canonical-label proximity, with a minimum score of _COL_MATCH_MIN_SCORE.
        """
        from rapidfuzz import fuzz

        out: dict[int, FieldSpec] = {}
        for col_idx, tokens in enumerate(header_tokens_per_col):
            header_text = " ".join(t.text for t in tokens).strip().lower()
            if not header_text:
                continue

            best_field: FieldSpec | None = None
            best_score: float = 0.0

            for field in line_fields:
                labels = field.canonical_labels or [field.name.replace("_", " ")]
                for label in labels:
                    score = float(fuzz.partial_ratio(header_text, label.lower()))
                    if score > best_score:
                        best_score = score
                        best_field = field

            if best_field is not None and best_score >= _COL_MATCH_MIN_SCORE:
                out[col_idx] = best_field

        return out
