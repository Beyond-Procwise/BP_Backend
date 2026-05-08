"""Tests for the extractive QA gap-filler extractor (Task 13 of the extraction-redesign plan).

Uses deepset/roberta-base-squad2 to produce answer spans for schema fields
that opt into "qa_roberta". Spans are substrings of full_text by construction;
we also re-verify this defensively.

Fixture: INV-007-rich.pdf ("MASTER Invoice for PO1") — the richest fixture in
the suite, giving QA the best chance of finding confident answers.

Run (requires GPU, model downloads ~500 MB on first use):
    .venv/bin/pytest tests/extraction_v3/test_qa_roberta.py -v -m gpu
"""
from pathlib import Path
import pytest
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.extractors.qa_roberta import QARobertaExtractor
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema


FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_qa_roberta_emits_substring_answer():
    """QA returns answer spans for fields it can answer; spans are substrings of full_text."""
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    ex = QARobertaExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    # On a rich invoice with multiple labels, QA should answer at least one
    assert candidates, "QA produced no answers"
    for c in candidates:
        assert c.evidence_text in parsed.full_text, f"answer not a substring: {c.evidence_text!r}"
        assert c.model == "qa_roberta"
        # Confidence above threshold
        assert c.confidence >= 0.4
