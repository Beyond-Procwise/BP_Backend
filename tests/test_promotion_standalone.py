"""Promotion of parent-less docs to _trgt.

Root cause of the stg->trgt stall: linking_engine._evaluate held EVERY invoice
or quote whose po_id was NULL with reason ``no_parent_reference`` and never
promoted it. But a quote is the ROOT of the quote->PO->invoice chain (no parent
by definition) and a no-PO invoice is valid off-PO / maverick spend. Such docs
must promote standalone when extraction confidence passes; there is no link
score to apply because there is no parent to score against.

Docs that DO reference a PO must keep the existing gates (low_link_score /
parent_not_found) — those are the correct HITL/timing holds and are unchanged.
"""
from __future__ import annotations
import src.services.linking_engine as le


class _FakeCur:
    """Minimal cursor; the no-parent branch never issues a query."""
    description: list = []
    def execute(self, *a, **k): self.description = []; self._r = []
    def fetchall(self): return []
    def fetchone(self): return None


def test_quote_without_po_is_promotable():
    # A quote has no parent PO by definition -> promotable when confidence passes.
    po, link, reason = le._evaluate(
        _FakeCur(), "quote", {"quote_id": "Q1", "po_id": None, "confidence_score": 100})
    assert reason is None, f"expected promotable, got {reason!r}"
    assert po is None and link is None


def test_no_po_invoice_is_promotable():
    # A no-PO (off-PO / maverick) invoice is valid -> promotable when conf passes.
    po, link, reason = le._evaluate(
        _FakeCur(), "invoice", {"invoice_id": "INV1", "po_id": None, "confidence_score": 95})
    assert reason is None, f"expected promotable, got {reason!r}"


def test_no_po_low_confidence_is_held():
    # No parent AND low extraction confidence -> held on confidence, not promoted.
    po, link, reason = le._evaluate(
        _FakeCur(), "quote", {"quote_id": "Q2", "po_id": None, "confidence_score": 50})
    assert reason == "low_extraction_confidence", f"got {reason!r}"


def test_linked_low_score_still_held(monkeypatch):
    # Regression guard: a doc that DOES reference a PO keeps the link-score gate.
    monkeypatch.setattr(le, "_find_parent_po", lambda c, p: {"po_id": "PO1"})
    monkeypatch.setattr(le, "_rows", lambda c, s, a=(): [])
    monkeypatch.setattr(le, "_set_amount_for_invoice", lambda c, p: None)
    monkeypatch.setattr(le, "score_link", lambda *a, **k: {
        "F": 10.0, "decision": "weak", "signals": {}, "P_raw": 0.0, "C": 0.0, "Q": 0.0})
    po, link, reason = le._evaluate(
        _FakeCur(), "invoice", {"invoice_id": "INV9", "po_id": "PO1", "confidence_score": 95})
    assert reason == "low_link_score", f"linked path changed: {reason!r}"
