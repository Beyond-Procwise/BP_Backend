"""Plant known problems in real deals, in memory only, and measure how many are caught.

Source data is never modified: each plant works on a deep copy of a loaded deal.
Run with PROCWISE_TEST_LIVE_DB=1. Prints a recall table (run with -s to see it).
"""
import copy
import dataclasses
import os
from decimal import Decimal as D

import pytest

from src.services.db import get_conn
from src.services.triage.engine import triage_set
from src.services.triage.link import link
from src.services.triage.loader import list_deal_ids, load_deal_sets
from src.services.triage.model import Line
from tests.triage.helpers import make_cfg

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")
CFG = make_cfg()


@pytest.fixture(scope="module")
def sample():
    with get_conn() as conn:
        cur = conn.cursor()
        ids = list_deal_ids(cur)[:200]
        return list(load_deal_sets(cur, ids).values())


def _target(ds):
    """First exactly-linked, non-credit invoice line with a priced PO line."""
    for lk in link(ds, CFG).line_links:
        if (lk.po_line is not None and lk.confidence == 1.0 and not lk.rollup
                and not lk.invoice.is_credit_note and lk.po_line.unit_price
                and lk.inv_line.unit_price and lk.inv_line.quantity and lk.po_line.quantity):
            return lk
    return None


def _all_results(out):
    return [r for f in out.findings for r in (*f.causes, *f.effects)]


def _plant_price(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, unit_price=old.unit_price * D("1.10"))
    return lambda out: any(r.rule_id == "unit_price" and r.claim_doc == inv.doc_id
                           and r.claim_line == old.line_ref for r in _all_results(out))


def _plant_quantity(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, quantity=old.quantity + lk.po_line.quantity)
    return lambda out: any(r.rule_id == "quantity" and r.po_id == lk.po.doc_id
                           and r.auth_line == lk.po_line.line_ref for r in _all_results(out))


def _plant_rebill(ds, lk):
    src = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    billed = sum((i.net or D("0")) for i in ds.invoices if i.po_ref == lk.po.doc_id)
    if lk.po.net is None or src.net is None or billed + src.net <= lk.po.net * D("1.01"):
        return None   # a re-bill here would not exceed the PO, so there is nothing to catch
    copy_ = copy.deepcopy(src)
    copy_.doc_id = f"{src.doc_id}-PLANTED"
    ds.invoices.append(copy_)
    return lambda out: any(r.rule_id in ("quantity", "cumulative_total")
                           and r.po_id == lk.po.doc_id for r in _all_results(out))


def _plant_currency(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.currency = "XXX"
    return lambda out: any(r.rule_id == "currency" and r.claim_doc == inv.doc_id
                           for r in _all_results(out))


def _plant_unlinked(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.lines.append(Line(line_ref="PLANTED", item_id="PLANTED-ITEM",
                          description="zzqx planted qqzz", quantity=D("1"),
                          unit_price=D("100"), line_amount=D("100")))
    return lambda out: any(r.rule_id == "unlinked_line" and r.claim_line == "PLANTED"
                           for r in _all_results(out))


PLANTS = {"price +10%": _plant_price, "quantity doubled": _plant_quantity,
          "invoice re-billed": _plant_rebill, "currency swapped": _plant_currency,
          "line with no PO line": _plant_unlinked}


def test_planted_problems_are_caught(sample):
    table = {}
    for name, plant in PLANTS.items():
        caught = tried = 0
        for original in sample:
            ds = copy.deepcopy(original)
            lk = _target(ds)
            if lk is None:
                continue
            detected = plant(ds, lk)
            if detected is None:
                continue
            tried += 1
            caught += bool(detected(triage_set(ds, CFG)))
        table[name] = (caught, tried)
    print("\nPlanted-error recall:")
    for name, (caught, tried) in table.items():
        print(f"  {name:22s} {caught}/{tried} = {caught / tried:.1%}")
    for name, (caught, tried) in table.items():
        assert tried >= 50, f"{name}: too few eligible deals ({tried})"
        assert caught / tried >= 0.95, f"{name}: recall {caught}/{tried}"
