"""The loader's contract leg, against a scripted cursor.

bp_contracts is empty in both databases, so a live test would prove only that an empty
table loads nothing. This scripts the rows the real queries would return and pins the
wiring: the contract a document cites is fetched, its amendment family comes with it,
its rate-card lines are attached, and a deal citing no contract issues no query.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest

from src.services.triage import loader
from src.services.triage.loader import load_deal_sets


class ScriptedCursor:
    """Answers each query by matching a distinctive fragment of its SQL."""

    def __init__(self, rows: dict[str, list[tuple]]):
        self.rows, self._last, self.seen = rows, [], []

    def execute(self, sql, params=None):
        for fragment, rows in self.rows.items():
            if fragment in sql:
                self.seen.append(fragment)
                self._last = list(rows)
                return
        self._last = []

    def fetchall(self):
        return self._last

    def fetchone(self):
        return self._last[0] if self._last else None


@pytest.fixture(autouse=True)
def _no_fx(monkeypatch):
    """FX is resolved per currency and is not what these tests are about."""
    class Rate:
        rate, rate_date = D("1"), None
    monkeypatch.setattr(loader, "resolve_fx", lambda *a, **k: Rate())


INVOICE = ("INV-1", "DEAL-1", "PO-1", "SUP-1", "GBP", date(2026, 2, 1),
           D("120"), D("24"), D("144"), "Net 30", 0.98, "C-1")
INVOICE_LINE = ("INV-1", "1", "ITEM-1", "Widget", D("10"), "each", D("12"), D("120"),
                None, None)
PO = ("PO-1", "DEAL-1", "SUP-1", "GBP", date(2026, 1, 10), D("100"), D("20"), D("120"),
      "Net 30", None, 0.99, None, None, "C-1")
PO_LINE = ("PO-1", "1", "ITEM-1", "Widget", D("10"), "each", D("10"), D("100"))


def _script(contracts, contract_lines, invoice=INVOICE):
    return {
        "FROM proc.bp_invoice_trgt": [invoice],
        "FROM proc.bp_invoice_line_items_trgt": [INVOICE_LINE],
        "FROM proc.bp_purchase_order_trgt": [PO],
        "FROM proc.bp_po_line_items_trgt": [PO_LINE],
        "FROM proc.bp_quote_trgt": [],
        "FROM proc.bp_quote_line_items_trgt": [],
        "FROM proc.bp_contracts": contracts,
        "FROM proc.bp_contract_line_items": contract_lines,
        "FROM proc.bp_extraction_discrepancy": [],
    }


def test_the_cited_contract_and_its_rate_card_are_loaded():
    cur = ScriptedCursor(_script(
        contracts=[("C-1", None, "SUP-1", "GBP", date(2026, 1, 1), "Net 30")],
        contract_lines=[("C-1", "1", "Widget", D("10.00"), "each", "cap", "price cap")]))
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    assert [c.doc_id for c in ds.contracts] == ["C-1"]
    term = ds.contracts[0].lines[0]
    assert (term.description, term.unit_price, term.term_basis) == ("Widget", D("10.00"), "cap")
    assert ds.invoices[0].contract_ref == "C-1"
    assert ds.pos[0].contract_ref == "C-1"


def test_the_amendment_family_comes_with_the_cited_contract():
    """The rate card may live on the parent or on a sibling amendment."""
    cur = ScriptedCursor(_script(
        contracts=[("C-1", None, "SUP-1", "GBP", date(2026, 1, 1), "Net 30"),
                   ("C-1-A1", "C-1", "SUP-1", "GBP", date(2026, 6, 1), "Net 30")],
        contract_lines=[("C-1-A1", "1", "Widget", D("9.00"), "each", "cap", None)]))
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    assert [c.doc_id for c in ds.contracts] == ["C-1", "C-1-A1"]
    assert ds.contracts[1].parent_contract_id == "C-1"
    assert ds.contracts[1].lines[0].unit_price == D("9.00")


def test_a_deal_citing_no_contract_runs_no_contract_query():
    """Two queries per batch is two too many for 5,000 deals that cite nothing."""
    no_ref = INVOICE[:11] + (None,)
    script = _script(contracts=[], contract_lines=[], invoice=no_ref)
    po_no_ref = PO[:13] + (None,)
    script["FROM proc.bp_purchase_order_trgt"] = [po_no_ref]
    cur = ScriptedCursor(script)
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    assert ds.contracts == []
    assert "FROM proc.bp_contracts" not in cur.seen
    assert "FROM proc.bp_contract_line_items" not in cur.seen


def test_a_cited_contract_that_is_gone_loads_nothing_rather_than_failing():
    cur = ScriptedCursor(_script(contracts=[], contract_lines=[]))
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    assert ds.contracts == []
    assert "FROM proc.bp_contract_line_items" not in cur.seen


def test_a_contract_with_no_lines_is_still_loaded():
    """A header-only contract is what every contract was before the rate card existed."""
    cur = ScriptedCursor(_script(
        contracts=[("C-1", None, "SUP-1", "GBP", date(2026, 1, 1), "Net 30")],
        contract_lines=[]))
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    assert [c.doc_id for c in ds.contracts] == ["C-1"]
    assert ds.contracts[0].lines == []


def test_the_loaded_contract_drives_the_check_end_to_end():
    """Loader -> link -> checks, so the wiring is proven joined up, not just present."""
    from src.services.triage.engine import triage_set
    from tests.triage.helpers import make_cfg
    cur = ScriptedCursor(_script(
        contracts=[("C-1", None, "SUP-1", "GBP", date(2026, 1, 1), "Net 30")],
        contract_lines=[("C-1", "1", "Widget", D("10.00"), "each", "cap", "price cap")]))
    ds = load_deal_sets(cur, ["DEAL-1"])["DEAL-1"]
    out = triage_set(ds, make_cfg())
    caps = [f for f in out.findings if f.rule_id == "contract_cap"]
    assert len(caps) == 1, "the invoice charges 12.00 against a 10.00 cap"
    assert caps[0].exposure == D("20.00")
