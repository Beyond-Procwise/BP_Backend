"""PO revisions in deal assignment and linking (Document match build 4, follow-up).

A revision ("4500018832 (Rev 3)") is the same purchase order re-issued. Invoices print the
bare number, so every revision must share it -- one deal, one anchoring quote, one order
budget -- and wherever a single PO is picked, it is the revision the citation means.
"""
from __future__ import annotations

import os

import pytest

from src.services import deal_assignment_service as das
from src.services import link_proposals as lp
from src.services.extraction.po_revision import pick_key
from src.services.linking_engine import _norm_po, _norm_po_id, cmp_po_ref
from tests.services.test_deal_assignment_service import (
    _anchor_batch, _anchor_ids, _table_scorer)


def test_a_revision_shares_its_po_number():
    assert _norm_po("PO 4500018832 (Rev 3)") == _norm_po("4500018832") == "4500018832"
    assert cmp_po_ref("4500018832", "4500018832 (Rev 3)") == (1.0, "OK")
    # ...while each row keeps its own identity
    assert _norm_po_id("4500018832 (Rev 3)") != _norm_po_id("4500018832")


def _row(po_id, rev=None, approval=None):
    return {"po_id": po_id, "po_revision": rev, "approval_status": approval}


def _pick(rows, cited):
    return max(rows, key=lambda r: pick_key(r, cited))["po_id"]


def test_a_bare_citation_means_the_latest_approved_revision():
    rows = [_row("X"), _row("X (Rev 2)", 2, "approved"), _row("X (Rev 3)", 3, "pending")]
    assert _pick(rows, "X") == "X (Rev 2)"
    assert _pick(rows, "PO X") == "X (Rev 2)"


def test_a_citation_naming_its_revision_keeps_it():
    rows = [_row("X"), _row("X (Rev 2)", 2, "approved"), _row("X (Rev 3)", 3, "pending")]
    assert _pick(rows, "X (Rev 3)") == "X (Rev 3)"


def test_with_no_approved_revision_the_cited_row_stands():
    rows = [_row("X", 1, "pending"), _row("X (Rev 2)", 2, "rejected")]
    assert _pick(rows, "X") == "X"


def test_revisions_share_one_anchoring_quote(monkeypatch):
    """Before, the second revision was refused the quote as 'one quote, two orders' and
    the re-issued PO was orphaned from its own sourcing event."""
    pos = [{"po_id": "P1", "supplier_id": "SUP-T", "supplier_name": "T", "deal_id": None},
           {"po_id": "P1 (Rev 2)", "supplier_id": "SUP-T", "supplier_name": "T", "deal_id": None},
           {"po_id": "P2", "supplier_id": "SUP-T", "supplier_name": "T", "deal_id": None}]
    quotes = [{"quote_id": "Q1", "po_id": None, "supplier_id": "SUP-T", "deal_id": None}]
    monkeypatch.setattr(das, "score_link", _table_scorer(
        {("P1", "Q1"): 95.0, ("P1 (Rev 2)", "Q1"): 95.0, ("P2", "Q1"): 90.0}))
    cur = _anchor_batch(pos, quotes)

    anchors = das._quote_anchors(cur, pos, cache=das._AnchorCache(cur))

    # the family shares Q1; a genuinely different order still cannot have it
    assert _anchor_ids(anchors) == {"P1": "Q1", "P1 (Rev 2)": "Q1", "P2": None}


def test_a_declared_quote_anchors_every_revision(monkeypatch):
    pos = [{"po_id": "P1", "supplier_id": "SUP-T", "supplier_name": "T", "deal_id": None},
           {"po_id": "P1 (Rev 3)", "supplier_id": "SUP-T", "supplier_name": "T", "deal_id": None}]
    quotes = [{"quote_id": "Q1", "po_id": "P1", "supplier_id": "SUP-T", "deal_id": None}]
    monkeypatch.setattr(das, "score_link", _table_scorer({}))
    cur = _anchor_batch(pos, quotes)

    anchors = das._quote_anchors(cur, pos, cache=das._AnchorCache(cur))

    assert _anchor_ids(anchors) == {"P1": "Q1", "P1 (Rev 3)": "Q1"}


def test_an_order_budget_counts_only_the_live_revision():
    orders = [_row("X"), _row("X (Rev 2)", 2, "approved"), _row("X (Rev 3)", 3, "pending"),
              _row("Y")]
    assert [o["po_id"] for o in lp._live_revisions(orders)] == ["X (Rev 2)", "Y"]


@pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                    reason="needs the .env database (PROCWISE_TEST_LIVE_DB=1)")
def test_find_parent_po_picks_across_tiers_on_the_live_schema():
    """A newer revision still in _stg outranks an older one already in _trgt."""
    from src.services.db import get_conn
    from src.services.linking_engine import _find_parent_po

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO proc.bp_purchase_order_trgt (po_id, total_amount) "
                        "VALUES ('DMT-7701', 100)")
            cur.execute("INSERT INTO proc.bp_purchase_order_stg (po_id, po_revision, "
                        "approval_status, total_amount) VALUES "
                        "('DMT-7701 (Rev 2)', 2, 'approved', 120), "
                        "('DMT-7701 (Rev 3)', 3, 'pending', 150)")
            assert _find_parent_po(cur, "PO DMT-7701")["po_id"] == "DMT-7701 (Rev 2)"
            assert _find_parent_po(cur, "DMT-7701 (Rev 3)")["po_id"] == "DMT-7701 (Rev 3)"
            assert _find_parent_po(cur, "DMT-7702") is None
        finally:
            cur.execute("DELETE FROM proc.bp_purchase_order_stg WHERE po_id LIKE 'DMT-7701%%'")
            cur.execute("DELETE FROM proc.bp_purchase_order_trgt WHERE po_id LIKE 'DMT-7701%%'")
