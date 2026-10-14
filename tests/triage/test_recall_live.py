"""Plant known problems in real deals, in memory only, and measure how many are caught.

Source data is never modified: each plant works on a deep copy of a loaded deal. A
planted change is credited as "caught" only when it actually moved the finding the
plant should have produced -- not merely when some result of the right rule_id/line
exists, since the seeded corpus is already heavily over-invoiced and such a result can
be present on the UNPLANTED deal too. Each plant therefore also runs the engine on an
unmodified copy of the same deal (the baseline) and the "caught" predicate compares the
planted run against that baseline.

Run with PROCWISE_TEST_LIVE_DB=1. Prints a recall table (run with -s to see it).
"""
import copy
import dataclasses
import os
from decimal import Decimal as D
from typing import Callable, NamedTuple, Optional

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


def _dec(value) -> Optional[D]:
    return None if value is None else D(value)


class Plant(NamedTuple):
    """raw(out): the old, non-discriminating check -- some result of the right rule_id
    and line exists on that document. It is kept only to report how often that shape is
    already present pre-plant (informational).

    caught(baseline_out, planted_out): the real test -- true only when the planted
    change actually moved the relevant value the way that plant should move it.
    """
    raw: Callable
    caught: Callable


def _plant_price(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, unit_price=old.unit_price * D("1.10"))

    def raw(out):
        return any(r.rule_id == "unit_price" and r.claim_doc == inv.doc_id
                   and r.claim_line == old.line_ref for r in _all_results(out))

    def caught(baseline_out, planted_out):
        return not raw(baseline_out) and raw(planted_out)

    return Plant(raw, caught)


def _plant_quantity(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, quantity=old.quantity + lk.po_line.quantity)
    increment = lk.po_line.quantity

    def raw(out):
        return any(r.rule_id == "quantity" and r.po_id == lk.po.doc_id
                   and r.auth_line == lk.po_line.line_ref for r in _all_results(out))

    def _quantity_result(out):
        return next((r for r in out.results if r.rule_id == "quantity"
                    and r.po_id == lk.po.doc_id and r.auth_line == lk.po_line.line_ref), None)

    def caught(baseline_out, planted_out):
        base = _quantity_result(baseline_out)
        base_cum = _dec(base.claim_value) if base is not None else D("0")
        expected = base_cum + increment
        return any(r.rule_id == "quantity" and r.po_id == lk.po.doc_id
                   and r.auth_line == lk.po_line.line_ref and r.claim_value is not None
                   and _dec(r.claim_value) == expected for r in _all_results(planted_out))

    return Plant(raw, caught)


def _plant_rebill(ds, lk):
    src = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    billed = sum((i.net or D("0")) for i in ds.invoices if i.po_ref == lk.po.doc_id)
    if lk.po.net is None or src.net is None or billed + src.net <= lk.po.net * D("1.01"):
        return None   # a re-bill here would not exceed the PO, so there is nothing to catch
    copy_ = copy.deepcopy(src)
    copy_.doc_id = f"{src.doc_id}-PLANTED"
    ds.invoices.append(copy_)
    added_net = src.net

    def raw(out):
        return any(r.rule_id in ("quantity", "cumulative_total") and r.po_id == lk.po.doc_id
                   for r in _all_results(out))

    def _cumulative_result(out):
        return next((r for r in out.results if r.rule_id == "cumulative_total"
                    and r.po_id == lk.po.doc_id), None)

    def _quantity_by_line(out):
        return {r.auth_line: _dec(r.claim_value) for r in out.results
                if r.rule_id == "quantity" and r.po_id == lk.po.doc_id and r.claim_value is not None}

    def caught(baseline_out, planted_out):
        base_cum = _cumulative_result(baseline_out)
        base_total = _dec(base_cum.claim_value) if base_cum is not None else D("0")
        expected = base_total + added_net
        cum_hit = any(r.rule_id == "cumulative_total" and r.po_id == lk.po.doc_id
                      and r.claim_value is not None and _dec(r.claim_value) == expected
                      for r in _all_results(planted_out))
        if cum_hit:
            return True
        base_qty = _quantity_by_line(baseline_out)
        return any(r.rule_id == "quantity" and r.po_id == lk.po.doc_id and r.claim_value is not None
                   and r.auth_line in base_qty and _dec(r.claim_value) > base_qty[r.auth_line]
                   for r in _all_results(planted_out))

    return Plant(raw, caught)


def _plant_currency(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.currency = "XXX"

    def raw(out):
        return any(r.rule_id == "currency" and r.claim_doc == inv.doc_id for r in _all_results(out))

    def caught(baseline_out, planted_out):
        return not raw(baseline_out) and raw(planted_out)

    return Plant(raw, caught)


def _plant_unlinked(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.lines.append(Line(line_ref="PLANTED", item_id="PLANTED-ITEM",
                          description="zzqx planted qqzz", quantity=D("1"),
                          unit_price=D("100"), line_amount=D("100")))

    def raw(out):
        return any(r.rule_id == "unlinked_line" and r.claim_line == "PLANTED"
                   for r in _all_results(out))

    def caught(baseline_out, planted_out):
        return not raw(baseline_out) and raw(planted_out)

    return Plant(raw, caught)


PLANTS = {"price +10%": _plant_price, "quantity doubled": _plant_quantity,
          "invoice re-billed": _plant_rebill, "currency swapped": _plant_currency,
          "line with no PO line": _plant_unlinked}


def test_planted_problems_are_caught(sample):
    table = {}
    missed = {}
    baseline_cache: dict = {}

    def _baseline(ds):
        out = baseline_cache.get(ds.deal_id)
        if out is None:
            out = triage_set(copy.deepcopy(ds), CFG)
            baseline_cache[ds.deal_id] = out
        return out

    for name, plant in PLANTS.items():
        caught = tried = already_flagged = 0
        missed_ids = []
        for original in sample:
            ds = copy.deepcopy(original)
            lk = _target(ds)
            if lk is None:
                continue
            plant_check = plant(ds, lk)
            if plant_check is None:
                continue
            tried += 1
            baseline_out = _baseline(original)
            planted_out = triage_set(ds, CFG)
            if plant_check.raw(baseline_out):
                already_flagged += 1
            hit = plant_check.caught(baseline_out, planted_out)
            caught += bool(hit)
            if not hit:
                missed_ids.append(original.deal_id)
        table[name] = (caught, tried, already_flagged)
        missed[name] = missed_ids

    print("\nPlanted-error recall:")
    for name, (caught, tried, already_flagged) in table.items():
        print(f"  {name:22s} {caught}/{tried} = {caught / tried:.1%}"
              f"   (already flagged pre-plant: {already_flagged}/{tried})")
        if missed[name]:
            print(f"    missed: {missed[name][:10]}")
    for name, (caught, tried, already_flagged) in table.items():
        assert tried >= 50, f"{name}: too few eligible deals ({tried})"
        assert caught / tried >= 0.95, (
            f"{name}: recall {caught}/{tried}; missed deals: {missed[name][:10]}")
