"""Reconciliation compares the purchase chain: the winning bid's current version, its PO,
its invoices -- never rival bids or superseded versions (2026-10-08)."""
from src.services.reconciliation import _chain_quotes, _flatten_docs


def _ctx(quotes, pos=(), invoices=()):
    return {"documents": {"quotes": list(quotes), "purchase_orders": list(pos), "invoices": list(invoices)}}


Q = [{"quote_id": "A-1", "supplier_id": "SA"}, {"quote_id": "A-1 (V2)", "supplier_id": "SA"},
     {"quote_id": "B-1", "supplier_id": "SB"}, {"quote_id": "B-1 (V3)", "supplier_id": "SB"}]


def test_the_awarded_suppliers_current_quote_is_the_chain():
    ids = [q["quote_id"] for q in _chain_quotes(_ctx(Q, pos=[{"po_id": "P1", "supplier_id": "SB"}]))]
    assert ids == ["B-1 (V3)"]


def test_competing_bids_with_no_award_are_not_reconciled_against_each_other():
    assert _chain_quotes(_ctx(Q)) == []


def test_a_single_bid_is_the_chain_at_its_latest_version():
    ids = [q["quote_id"] for q in _chain_quotes(_ctx(Q[:2]))]
    assert ids == ["A-1 (V2)"]


def test_flatten_uses_the_chain_quotes_only():
    docs = _flatten_docs(_ctx(Q, pos=[{"po_id": "P1", "supplier_id": "SB"}]))
    assert [(d["doc_kind"], d["doc_pk"]) for d in docs] == [("purchase_order", "P1"), ("quote", "B-1 (V3)")]
