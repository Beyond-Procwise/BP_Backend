"""Tests for the Table Transformer extractor (Task 10 of the extraction-redesign plan).

Uses microsoft/table-transformer-structure-recognition-v1.1-all on rasterized
document pages to detect table structure (rows, columns, header) and produce
line_items[N].field_name Candidates.

Fixture: INV-007-rich.pdf — a multi-row invoice with explicit table structure
(Description / Qty / Unit Price / Amount columns). Selected over INV-001-clean.pdf
because it has a proper line-item table with at least 2 rows and column headers
that align with the invoice.yaml canonical_labels.

Run:
    .venv/bin/pytest tests/extraction_v3/test_table_transformer.py -v
GPU (CUDA) is required — the model is loaded onto cuda:0.
"""
from pathlib import Path

import pytest

from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.extractors.table_transformer import TableTransformerExtractor
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_table_transformer_emits_line_item_candidates():
    """Pre-trained Table Transformer extracts table structure from a clean invoice.

    Expectations:
      - At least one line_items[N].* candidate is produced per data row.
      - All field names match the pattern ``line_items[N].field_name``.
      - Every candidate's evidence_text is a substring of parsed.full_text
        (substring guarantee).
      - At least one candidate for a common field type (description or line_amount).
      - All confidence values are in [0, 1].
      - All candidates carry model='table_transformer'.
    """
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    extractor = TableTransformerExtractor()
    candidates = extractor.produce_candidates(parsed, schema)

    # ---------------------------------------------------------------------- #
    # Basic liveness                                                           #
    # ---------------------------------------------------------------------- #
    assert candidates, (
        "Table Transformer produced no line-item candidates. "
        "Check GPU availability, DETECTION_THRESHOLD, and that INV-007-rich.pdf "
        "has a structured table with column headers."
    )

    # ---------------------------------------------------------------------- #
    # Field name pattern                                                       #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.field.startswith("line_items["), (
            f"Unexpected field name: {c.field!r} — expected line_items[N].field"
        )
        # Must also have a dot-separated field part
        assert "." in c.field, (
            f"Field {c.field!r} missing '.field_name' suffix"
        )

    # ---------------------------------------------------------------------- #
    # Substring guarantee — critical for hallucination detection (Task 19)    #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.evidence_text in parsed.full_text, (
            f"Candidate evidence_text {c.evidence_text!r} for field {c.field!r} "
            f"is NOT a substring of full_text — substring guarantee broken.\n"
            f"full_text[:300] = {parsed.full_text[:300]!r}"
        )

    # ---------------------------------------------------------------------- #
    # Field type coverage — at least one schema line-item field present        #
    # ---------------------------------------------------------------------- #
    # Accept any valid invoice.yaml line_items field name
    valid_line_item_fields = {
        "item_description", "quantity", "unit_price",
        "line_amount", "tax_percent", "tax_amount", "total_amount_incl_tax",
        # canonical aliases the mapper may use
        "description", "amount",
    }
    field_types = {c.field.split(".")[-1] for c in candidates}
    assert field_types & valid_line_item_fields, (
        f"No recognised line-item field candidates found. Got field types: {field_types}. "
        f"Expected at least one of: {valid_line_item_fields}"
    )

    # ---------------------------------------------------------------------- #
    # Confidence bounds                                                        #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert 0.0 <= c.confidence <= 1.0, (
            f"Confidence out of bounds: {c.confidence} for {c.field!r}"
        )

    # ---------------------------------------------------------------------- #
    # Model tag                                                                #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.model == "table_transformer", (
            f"Expected model='table_transformer', got {c.model!r}"
        )

    # ---------------------------------------------------------------------- #
    # Value is non-empty                                                       #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.value.strip(), (
            f"Empty value for candidate {c.field!r}"
        )
