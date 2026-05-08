"""Tests for the sBERT semantic anchor extractor (Task 11 of the extraction-redesign plan).

The sbert_anchor extractor embeds label-shaped tokens and matches them to
schema fields via cosine similarity to the field's canonical_labels embeddings.
It catches synonymy that canonical_labels misses — e.g. 'Sold By:' →
supplier_name; 'Account Number:' → buyer_id.

Fixture: INV-007-rich.pdf ("MASTER Invoice for PO1") — same fixture used by
test_layoutlmv3_extractor.py, which reliably produces label tokens.

Run (GPU recommended, model downloads ~420 MB on first use):
    .venv/bin/pytest tests/extraction_v3/test_sbert_anchor.py -v -m gpu
"""
from pathlib import Path
import pytest
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.extractors.sbert_anchor import SbertAnchorExtractor
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_sbert_anchor_emits_candidates_with_substring_evidence():
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    ex = SbertAnchorExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    # On a rich invoice with multiple labels, sBERT should produce at least 1 candidate
    assert candidates, "sBERT produced no candidates"
    # Substring guarantee
    for c in candidates:
        assert c.evidence_text in parsed.full_text, f"evidence missing from full_text: {c.evidence_text!r}"
        assert c.model == "sbert_anchor"
        assert 0.0 <= c.confidence <= 1.0
        # Field should be one that opted in to sbert_anchor in invoice.yaml
        sbert_fields = {f.name for f in schema.fields if "sbert_anchor" in f.extractors}
        assert c.field in sbert_fields, f"unexpected field {c.field}"
