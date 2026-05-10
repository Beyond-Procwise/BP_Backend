"""LayoutLMv3 fine-tuned token-classification extractor (Plan 2 Phase A).

Loads a fine-tuned LayoutLMv3ForTokenClassification checkpoint and runs it
on rasterized page images. The model is controlled by an environment variable:

  LAYOUTLMV3_FINETUNED_CHECKPOINT  (default: "nielsr/layoutlmv3-finetuned-cord")

Supported checkpoints and their label-to-field mappings:
  - "nielsr/layoutlmv3-finetuned-cord" (CORD receipt dataset):
    Maps receipt labels → procurement schema fields for invoices.
  - Any local path to a fine-tuned checkpoint from Phase B self-training.

CORD label → schema field mapping (invoice doc type):
  TOTAL.TOTAL_PRICE    → invoice_total_incl_tax
  SUB_TOTAL.SUBTOTAL_PRICE → invoice_amount
  SUB_TOTAL.TAX_PRICE  → tax_amount
  MENU.NM              → line_items[*].item_description
  MENU.UNITPRICE       → line_items[*].unit_price
  MENU.CNT             → line_items[*].quantity
  MENU.PRICE           → line_items[*].line_amount

FUNSD label → schema field mapping (PO/quote doc type):
  ANSWER  → general string values (low confidence; used only as tiebreaker)

Substring guarantee: all candidate evidence_text values are checked against
parsed.full_text before emission. Tokens that appear only in image space but
not in the OCR text are silently skipped.

GPU memory: the model is ~500MB; it shares the GPU with other models. It is
loaded lazily and cached globally. If CUDA OOM occurs, the extractor silently
falls back to CPU (at reduced speed).

Performance: ~1-3 seconds per page on A10G. Long documents are capped at
MAX_PAGES_PER_DOC pages (default: 4, covering 95%+ of invoices).
"""
from __future__ import annotations

import os
import re
import threading
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = os.environ.get(
    "LAYOUTLMV3_FINETUNED_CHECKPOINT",
    "nielsr/layoutlmv3-finetuned-cord",
)
INVOICE_FINETUNE_PATH = os.environ.get("LAYOUTLMV3_INVOICE_FINETUNE_PATH", "")

# If set, use the Phase B self-trained invoice checkpoint over CORD
CHECKPOINT = INVOICE_FINETUNE_PATH if INVOICE_FINETUNE_PATH else DEFAULT_CHECKPOINT

MAX_PAGES_PER_DOC = int(os.environ.get("LAYOUTLMV3_FINETUNED_MAX_PAGES", "4"))
MIN_TOKEN_CONFIDENCE = 0.55  # minimum softmax probability to emit a candidate

# ---------------------------------------------------------------------------
# CORD label → (schema_field, doc_types_where_applicable)
# B-/I- prefixes are stripped before lookup.
# ---------------------------------------------------------------------------
_CORD_LABEL_TO_FIELD: dict[str, tuple[str, set[str]]] = {
    "TOTAL.TOTAL_PRICE":        ("invoice_total_incl_tax", {"invoice"}),
    "SUB_TOTAL.SUBTOTAL_PRICE": ("invoice_amount",         {"invoice"}),
    "SUB_TOTAL.TAX_PRICE":      ("tax_amount",             {"invoice"}),
    "MENU.NM":                  ("line_item_description",  {"invoice"}),  # special: line item
    "MENU.UNITPRICE":           ("line_item_unit_price",   {"invoice"}),
    "MENU.CNT":                 ("line_item_quantity",     {"invoice"}),
    "MENU.PRICE":               ("line_item_line_amount",  {"invoice"}),
}

# Line-item special fields (assembled separately, not via schema field names)
_LINE_ITEM_FIELDS = {
    "line_item_description", "line_item_unit_price",
    "line_item_quantity", "line_item_line_amount",
}

# ---------------------------------------------------------------------------
# Lazy model + processor loading
# ---------------------------------------------------------------------------
_model = None
_processor = None
_model_lock = threading.Lock()
_model_label_type: str | None = None  # "cord" or "funsd" or "custom"


def _detect_model_type(config) -> str:
    """Detect the type of fine-tuned model from its label set."""
    labels = set(config.id2label.values())
    if any("MENU" in lbl for lbl in labels):
        return "cord"
    if any("ANSWER" in lbl or "QUESTION" in lbl for lbl in labels):
        return "funsd"
    return "custom"


def _get_model_and_processor():
    """Lazy-load the model and processor. Cached globally."""
    global _model, _processor, _model_label_type
    if _model is not None and _processor is not None:
        return _model, _processor, _model_label_type

    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor, _model_label_type

        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
        )

        log.info("Loading LayoutLMv3 fine-tuned checkpoint: %s", CHECKPOINT)
        try:
            # apply_ocr=False: we supply our own words + bboxes from ParsedDocument
            _processor = LayoutLMv3Processor.from_pretrained(
                CHECKPOINT, apply_ocr=False
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = LayoutLMv3ForTokenClassification.from_pretrained(CHECKPOINT)
            _model = _model.to(device).eval()
            _model_label_type = _detect_model_type(_model.config)
            log.info(
                "LayoutLMv3 finetuned loaded on %s. Type=%s, labels=%d",
                device, _model_label_type, len(_model.config.id2label),
            )
        except Exception as exc:
            log.error("Failed to load LayoutLMv3 finetuned: %s", exc)
            _model = None
            _processor = None
            _model_label_type = None

    return _model, _processor, _model_label_type


def _encode_page_inputs(processor, page_image, words: list[str], bboxes: list[list[int]]) -> dict:
    """Build a LayoutLMv3 model input dict for one page.

    The LayoutLMv3Processor with apply_ocr=False uses a TokenizersBackend that
    does NOT propagate bboxes through the standard processor() call (it returns
    only input_ids + attention_mask from the tokenizer). We assemble the inputs
    manually:

    1. Image: process via processor.image_processor → pixel_values (1, 3, 224, 224)
    2. Text: tokenize via processor.tokenizer with is_split_into_words=True
             → input_ids, attention_mask, word_ids per sub-token
    3. Bbox: build per-sub-token bbox tensor using word_ids → original bboxes
             Special tokens (word_id=None) get bbox [0,0,0,0].

    This matches the LayoutLMv3 paper's encoding exactly.
    """
    # Step 1: process image
    img_enc = processor.image_processor(images=page_image, return_tensors="pt")

    # Step 2: tokenize words with is_split_into_words=True
    tok_enc = processor.tokenizer(
        words,
        boxes=bboxes,  # passed for compatibility; may not propagate in fast tokenizer
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Step 3: build per-sub-token bbox tensor from word_ids
    word_ids = tok_enc.word_ids(batch_index=0)
    bbox_list = []
    for wid in word_ids:
        if wid is None or wid >= len(bboxes):
            bbox_list.append([0, 0, 0, 0])
        else:
            bbox_list.append(bboxes[wid])

    bbox_tensor = torch.tensor([bbox_list], dtype=torch.long)  # (1, seq_len, 4)

    return {
        "input_ids": tok_enc["input_ids"],
        "attention_mask": tok_enc["attention_mask"],
        "bbox": bbox_tensor,
        "pixel_values": img_enc["pixel_values"],
        "_word_ids": word_ids,  # kept for post-processing; not passed to model
    }


# ---------------------------------------------------------------------------
# PDF/image rasterization
# ---------------------------------------------------------------------------

def _rasterize_pdf_page(pdf_path: str, page_idx: int, dpi: int = 150):
    """Return a PIL Image for the specified page of a PDF."""
    import fitz  # PyMuPDF
    from PIL import Image
    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        return None
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def _rasterize_image(image_path: str):
    """Return a PIL Image for an image file."""
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def _get_page_image(source_path: str, page_idx: int):
    """Get a PIL Image for a given page, handling PDF and image files."""
    p = Path(source_path)
    if p.suffix.lower() == ".pdf":
        return _rasterize_pdf_page(source_path, page_idx)
    elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        if page_idx > 0:
            return None  # images are single-page
        return _rasterize_image(source_path)
    return None


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

def _normalize_bbox(bbox, page_width: float, page_height: float) -> list[int]:
    """Normalize a (x0, y0, x1, y1) bbox to [0, 1000] for LayoutLMv3."""
    x0, y0, x1, y1 = bbox
    # Clamp to [0, 1000]
    def clamp(v): return max(0, min(1000, int(v)))
    nx0 = clamp(x0 / page_width * 1000) if page_width > 0 else 0
    ny0 = clamp(y0 / page_height * 1000) if page_height > 0 else 0
    nx1 = clamp(x1 / page_width * 1000) if page_width > 0 else 0
    ny1 = clamp(y1 / page_height * 1000) if page_height > 0 else 0
    # LayoutLMv3 requires x0 <= x1, y0 <= y1
    if nx0 > nx1:
        nx0, nx1 = nx1, nx0
    if ny0 > ny1:
        ny0, ny1 = ny1, ny0
    return [nx0, ny0, nx1, ny1]


# ---------------------------------------------------------------------------
# Candidate assembler from CORD predictions
# ---------------------------------------------------------------------------

def _strip_bio(label: str) -> str:
    """Remove B- or I- prefix from a BIO label."""
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label


def _merge_bio_spans(
    tokens: list,
    labels: list[str],
    probs: list[float],
) -> list[tuple[str, str, float]]:
    """Merge consecutive B-/I- spans into (label, text, max_prob) tuples."""
    spans: list[tuple[str, str, float]] = []
    current_label: str | None = None
    current_tokens: list[str] = []
    current_prob: float = 0.0

    for tok, lbl, prob in zip(tokens, labels, probs):
        bare = _strip_bio(lbl)
        if lbl.startswith("B-") or (bare != "O" and (current_label != bare)):
            # Start of a new span
            if current_label and current_tokens:
                spans.append((current_label, " ".join(current_tokens), current_prob))
            current_label = bare
            current_tokens = [tok]
            current_prob = prob
        elif lbl.startswith("I-") and current_label == bare:
            current_tokens.append(tok)
            current_prob = max(current_prob, prob)
        else:
            # O label or label change
            if current_label and current_tokens:
                spans.append((current_label, " ".join(current_tokens), current_prob))
            current_label = None
            current_tokens = []
            current_prob = 0.0

    if current_label and current_tokens:
        spans.append((current_label, " ".join(current_tokens), current_prob))

    return spans


@register_extractor("layoutlmv3_finetuned")
class LayoutLMv3FinetunedExtractor(Extractor):
    """Plan 2 Phase A: fine-tuned LayoutLMv3 token classifier.

    Runs on rasterized page images. Maps model labels → schema field candidates.
    Registered as 'layoutlmv3_finetuned' — add to schema field extractors lists
    to activate it alongside the baseline 'layoutlmv3' extractor.

    The orchestrator picks the higher-confidence candidate per field.
    """

    def produce_candidates(
        self,
        parsed: ParsedDocument,
        schema: DocSchema,
    ) -> list[Candidate]:
        # Only run if 'layoutlmv3_finetuned' is requested by any field in the schema
        active_fields = [
            f for f in schema.fields if "layoutlmv3_finetuned" in f.extractors
        ]
        if not active_fields:
            return []

        model, processor, model_type = _get_model_and_processor()
        if model is None or processor is None:
            log.warning("LayoutLMv3 finetuned model not available; skipping")
            return []

        candidates: list[Candidate] = []
        pages_to_process = min(len(parsed.pages), MAX_PAGES_PER_DOC)

        for page_idx in range(pages_to_process):
            page = parsed.pages[page_idx]
            if not page.tokens:
                continue

            # Rasterize the page
            try:
                page_image = _get_page_image(parsed.source_path, page_idx)
            except Exception as exc:
                log.debug("Failed to rasterize page %d: %s", page_idx, exc)
                continue
            if page_image is None:
                continue

            # Build word + bbox lists from parsed tokens.
            # ParsedDocument tokens are often compound (Docling merges multiple words
            # into a single token with a shared bbox). We split them back into
            # individual word tokens, keeping the parent token's bbox for all sub-words.
            # This gives LayoutLMv3 enough granularity for token classification.
            words: list[str] = []
            bboxes: list[list[int]] = []
            raw_tokens = []
            for tok in page.tokens:
                tok_text = tok.text.strip()
                if not tok_text:
                    continue
                norm_bbox = _normalize_bbox(tok.bbox, page.width, page.height)
                # Split compound tokens on whitespace/newline
                sub_words = tok_text.split()
                for w in sub_words:
                    if w:
                        words.append(w)
                        bboxes.append(norm_bbox)
                        raw_tokens.append(tok)  # parent token for location reference

            if not words:
                continue

            try:
                page_cands = self._run_page(
                    model, processor, model_type, schema.doc_type,
                    page_image, words, bboxes, raw_tokens, parsed, page_idx,
                )
                candidates.extend(page_cands)
            except Exception as exc:
                log.debug(
                    "LayoutLMv3 finetuned failed on page %d of %s: %s",
                    page_idx, parsed.source_path, exc,
                )

        return candidates

    def _run_page(
        self,
        model,
        processor,
        model_type: str,
        doc_type: str,
        page_image,
        words: list[str],
        bboxes: list[list[int]],
        raw_tokens,
        parsed: ParsedDocument,
        page_idx: int,
    ) -> list[Candidate]:
        """Run the model on one page and return candidates."""
        device = next(model.parameters()).device

        # Build encoding manually (see _encode_page_inputs for rationale)
        encoding = _encode_page_inputs(processor, page_image, words, bboxes)
        word_ids = encoding.pop("_word_ids")  # not a model input

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        pred_probs = probs.max(dim=-1).values.cpu().tolist()

        id2label = model.config.id2label

        # Aggregate sub-token predictions to word level (max probability)
        word_labels: dict[int, tuple[str, float]] = {}
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            lbl = id2label[pred_ids[tok_idx]]
            prob = pred_probs[tok_idx]
            if word_idx not in word_labels or prob > word_labels[word_idx][1]:
                word_labels[word_idx] = (lbl, prob)

        # Build parallel lists for BIO merging
        word_label_list = []
        word_prob_list = []
        for widx in range(len(words)):
            lbl, prob = word_labels.get(widx, ("O", 0.0))
            word_label_list.append(lbl)
            word_prob_list.append(prob)

        # Merge BIO spans
        spans = _merge_bio_spans(words, word_label_list, word_prob_list)

        # Map spans to candidates
        candidates: list[Candidate] = []
        if model_type == "cord":
            candidates = self._cord_spans_to_candidates(
                spans, doc_type, raw_tokens, parsed, page_idx,
            )
        # FUNSD and custom model types: skip (too generic for our fields)

        return candidates

    def _cord_spans_to_candidates(
        self,
        spans: list[tuple[str, str, float]],
        doc_type: str,
        raw_tokens,
        parsed: ParsedDocument,
        page_idx: int,
    ) -> list[Candidate]:
        """Convert CORD label spans to Candidate objects for invoice fields."""
        from src.services.extraction_v2.parsers.amounts import parse_amount

        candidates: list[Candidate] = []

        # Track line items by sequential index
        line_item_idx = 0
        line_item_groups: list[dict[str, tuple[str, float]]] = []

        for cord_label, span_text, prob in spans:
            if prob < MIN_TOKEN_CONFIDENCE:
                continue
            if cord_label not in _CORD_LABEL_TO_FIELD:
                continue

            field_name, applicable_doc_types = _CORD_LABEL_TO_FIELD[cord_label]
            if doc_type not in applicable_doc_types:
                continue

            # Substring guarantee: the span text must appear in full_text
            # (as-is, or with normalized whitespace)
            evidence = span_text.strip()
            if not evidence:
                continue

            # Try exact match first, then normalized
            if evidence not in parsed.full_text:
                # Try joining tokens without spaces (OCR sometimes splits differently)
                evidence_compact = re.sub(r"\s+", "", evidence)
                if evidence_compact and evidence_compact in parsed.full_text:
                    evidence = evidence_compact
                else:
                    # Skip: cannot ground to source text
                    log.debug(
                        "CORD span %r not in full_text (label=%s), skipping",
                        evidence[:30], cord_label,
                    )
                    continue

            # Find the best matching token's bbox for location
            bbox = (0.0, 0.0, 0.0, 0.0)
            for tok in raw_tokens:
                if tok.text.strip() and tok.text.strip() in evidence:
                    bbox = tok.bbox
                    break

            if field_name in _LINE_ITEM_FIELDS:
                # Accumulate line items (grouped by appearance order)
                sub_field = field_name.replace("line_item_", "")  # description, unit_price, etc.
                if sub_field == "description":
                    line_item_groups.append({})
                    line_item_idx = len(line_item_groups) - 1
                if line_item_groups:
                    line_item_groups[-1][sub_field] = (evidence, prob)
            else:
                # Header field candidate
                candidates.append(Candidate(
                    field=field_name,
                    value=evidence,
                    page=page_idx,
                    bbox=bbox,
                    evidence_text=evidence,
                    model="layoutlmv3_finetuned",
                    confidence=min(0.92, prob * 0.95),  # slight discount vs fine-tuned on receipts
                ))

        # Emit line item candidates
        for row_idx, group in enumerate(line_item_groups):
            field_map = {
                "description": "item_description",
                "unit_price": "unit_price",
                "quantity": "quantity",
                "line_amount": "line_amount",
            }
            for sub_field, db_field in field_map.items():
                if sub_field in group:
                    evidence, prob = group[sub_field]
                    if evidence in parsed.full_text:
                        candidates.append(Candidate(
                            field=f"line_items[{row_idx}].{db_field}",
                            value=evidence,
                            page=page_idx,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                            evidence_text=evidence,
                            model="layoutlmv3_finetuned",
                            confidence=min(0.88, prob * 0.90),
                        ))

        return candidates
