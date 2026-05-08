"""Tests for the LayoutLMv3 extractor (Task 9 of the extraction-redesign plan).

PLAN 1: canonical-label proximity over Docling-parsed tokens.
No fine-tuned model heads are exercised here — the pre-trained
microsoft/layoutlmv3-base has no procurement labels.

Fixture choice: INV-007-rich.pdf ("MASTER Invoice for PO1") is used in
preference to INV-001-clean.pdf ("Family Fun") because it carries explicit
invoice-standard labels — "Invoice Number", "Invoice Date", "Due Date",
"Subtotal", "Total Amount" — that map directly to the invoice.yaml
canonical_labels and make the proximity matcher reliably produce candidates.
INV-001-clean.pdf uses non-standard labels ("Document Number", "Balance Due")
that pass a contains-check but are less reliable for threshold tuning.

Run with GPU fixture or on any machine with Docling's local models cached:
    .venv/bin/pytest tests/extraction_v3/test_layoutlmv3_extractor.py -v
Deselect on GPU-only runners:
    .venv/bin/pytest tests/extraction_v3/test_layoutlmv3_extractor.py -m "not gpu"
"""

from pathlib import Path

import pytest

from src.services.extraction_v3.extractors.layoutlmv3 import LayoutLMv3Extractor
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu  # Docling downloads HF models on first run; GPU optional but expected
def test_layoutlmv3_emits_candidates_with_cited_evidence():
    """LayoutLMv3 extractor produces candidates; every evidence_text is in full_text.

    With Plan 1 proximity matching on INV-007-rich.pdf the matcher should
    locate at least one header field (e.g. invoice_id from "Invoice Number:",
    or invoice_date from "Invoice Date:").  Real accuracy comes after Plan 2
    fine-tuning; this test just verifies the pipeline contract.
    """
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    extractor = LayoutLMv3Extractor()
    candidates = extractor.produce_candidates(parsed, schema)

    assert candidates, (
        "No candidates produced — proximity matcher misconfigured or INV-007-rich.pdf "
        "has no tokens matching the invoice.yaml canonical_labels. "
        f"Inspect parsed.full_text:\n{parsed.full_text[:400]}"
    )

    # ---------------------------------------------------------------------- #
    # Substring guarantee — critical for Task 19's hallucination check        #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.evidence_text in parsed.full_text, (
            f"Candidate evidence_text {c.evidence_text!r} for field {c.field!r} "
            f"is NOT a substring of full_text — substring guarantee broken.\n"
            f"full_text[:200] = {parsed.full_text[:200]!r}"
        )

    # ---------------------------------------------------------------------- #
    # Field membership — candidates must be for layoutlmv3-enabled fields     #
    # ---------------------------------------------------------------------- #
    lmv3_fields = {f.name for f in schema.fields if "layoutlmv3" in f.extractors}
    for c in candidates:
        assert c.field in lmv3_fields, (
            f"Candidate for field {c.field!r} but that field does not list "
            f"'layoutlmv3' in its extractors."
        )

    # ---------------------------------------------------------------------- #
    # Confidence bounds                                                        #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert 0.0 <= c.confidence <= 1.0, (
            f"Confidence out of bounds: {c.confidence} for field {c.field!r}"
        )
    # Plan 1 baseline caps at 0.95; no candidate should claim certainty
    for c in candidates:
        assert c.confidence < 1.0, (
            f"Pre-trained baseline must not emit confidence=1.0, got {c.confidence} "
            f"for field {c.field!r}"
        )

    # ---------------------------------------------------------------------- #
    # Model tag                                                                #
    # ---------------------------------------------------------------------- #
    for c in candidates:
        assert c.model == "layoutlmv3", (
            f"Expected model='layoutlmv3', got {c.model!r}"
        )
