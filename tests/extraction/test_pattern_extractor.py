"""Tests for the L1 pattern extractor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.pattern_extractor import run_pattern_extractor
from src.services.extraction.pattern_registry import clear_cache


class _ParsedStub:
    """Minimum object shape pattern_extractor needs: a `full_text` and `pages`."""

    def __init__(self, full_text: str) -> None:
        self.full_text = full_text
        self.pages = []


def setup_function():
    clear_cache()


def test_invoice_id_anchored_pattern():
    parsed = _ParsedStub("Invoice Number: INV-4837\nDate: 2026-04-12")
    cands = run_pattern_extractor(parsed, "invoice")
    inv = [c for c in cands if c.field == "invoice_id"]
    assert inv, "expected an invoice_id candidate"
    assert inv[0].value == "INV-4837"
    assert inv[0].source == "regex"
    assert inv[0].confidence >= 0.80
    # span.text must be substring of full_text (grounding invariant)
    assert inv[0].span.text in parsed.full_text


def test_invoice_amount_from_subtotal_label():
    parsed = _ParsedStub("Subtotal: £1,200.00\nTax: £240.00\nTotal: £1,440.00")
    cands = run_pattern_extractor(parsed, "invoice")
    amt = [c for c in cands if c.field == "invoice_amount"]
    assert amt
    assert amt[0].value == "1,200.00"
    assert amt[0].span.text in parsed.full_text


def test_supplier_name_anchored():
    parsed = _ParsedStub("Supplier: Acme Industrial Supplies Ltd\nVAT: GB123456")
    cands = run_pattern_extractor(parsed, "invoice")
    sup = [c for c in cands if c.field == "supplier_name"]
    assert sup
    assert "Acme Industrial Supplies Ltd" in sup[0].value


def test_no_match_yields_no_candidate():
    parsed = _ParsedStub("Random text with no procurement labels at all.")
    cands = run_pattern_extractor(parsed, "invoice")
    inv = [c for c in cands if c.field == "invoice_id"]
    assert inv == []


def test_substring_grounding_holds_for_all_candidates():
    text = "Invoice Number: INV-9999\nSupplier: TestCo Ltd\nSubtotal: £45.99"
    parsed = _ParsedStub(text)
    cands = run_pattern_extractor(parsed, "invoice")
    assert cands
    for c in cands:
        assert c.span.text in text, f"candidate {c} ungrounded"


def test_multiple_invoice_id_pattern_priority():
    """When both 'Invoice Number:' and the bareword INV-... appear, the
    anchored pattern (higher prior) wins."""
    parsed = _ParsedStub("Invoice Number: ABC-100\nReference: INV-99887766")
    cands = run_pattern_extractor(parsed, "invoice")
    inv = [c for c in cands if c.field == "invoice_id"]
    assert inv
    assert inv[0].value == "ABC-100"
