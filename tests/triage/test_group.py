from datetime import date
from decimal import Decimal as D

from src.services.triage.group import group
from src.services.triage.model import DuplicateFlag, Severity
from tests.triage.helpers import deal, inv, line, make_cfg, po, scored

CFG = make_cfg()


def _findings(ds):
    _links, results = scored(ds, CFG)
    return group(results, CFG)


def test_clean_deal_has_no_findings():
    assert _findings(deal(po(), inv())) == []


def test_price_error_is_one_finding_with_the_po_total_as_its_effect():
    ds = deal(po(lines=[line(1, qty="300", price="12.00")]),
              inv(lines=[line(1, qty="300", price="13.50")]))
    (f,) = _findings(ds)
    assert f.rule_id == "unit_price"
    assert [e.rule_id for e in f.effects] == ["cumulative_total"]
    assert f.severity == Severity.S1 and f.exposure == D("450")


def test_flagged_duplicate_absorbs_quantity_and_overbilling():
    ds = deal(po(), inv("INV-1", inv_date=date(2026, 2, 1)), inv("INV-2", inv_date=date(2026, 2, 2)),
              duplicates=[DuplicateFlag("INV-2", "INV-1", D("144"))])
    (f,) = _findings(ds)
    assert f.rule_id == "duplicate"
    assert sorted(e.rule_id for e in f.effects) == ["cumulative_total", "quantity"]
    assert f.severity == Severity.S1


def test_unflagged_rebill_is_one_quantity_finding():
    ds = deal(po(lines=[line(1), line(2, item="ITEM-2")]),
              inv("INV-1", lines=[line(1), line(2, item="ITEM-2")], inv_date=date(2026, 2, 1)),
              inv("INV-2", lines=[line(1), line(2, item="ITEM-2")], inv_date=date(2026, 2, 2)))
    (f,) = _findings(ds)
    assert f.rule_id == "quantity" and len(f.causes) == 2
    assert [e.rule_id for e in f.effects] == ["cumulative_total"]
    assert f.severity == Severity.S1          # the always-S1 effect is never softened


def test_uniform_uplift_groups_lines_with_the_same_percentage():
    items = "ABCD"
    ds = deal(po(lines=[line(i, item=items[i], qty="1", price="100.00") for i in range(4)]),
              inv(lines=[line(i, item=items[i], qty="1", price="103.50") for i in range(4)]))
    (f,) = _findings(ds)
    assert f.rule_id == "uniform_uplift" and len(f.causes) == 4


def test_mixed_percentages_below_the_group_size_stay_separate():
    prices = ["103.50", "103.50", "110.00"]
    ds = deal(po(lines=[line(i, item=str(i), qty="1", price="100.00") for i in range(3)]),
              inv(lines=[line(i, item=str(i), qty="1", price=p) for i, p in enumerate(prices)]))
    fs = _findings(ds)
    assert sorted(f.rule_id for f in fs) == ["unit_price"] * 3


def test_unexplained_part_of_an_overage_is_its_own_finding():
    ds = deal(po(), inv(lines=[line(1, price="12.50")], net="200"))
    fs = {f.rule_id: f for f in _findings(ds)}
    assert set(fs) == {"unit_price", "cumulative_total", "invoice_totals"}
    assert fs["cumulative_total"].exposure == D("75")
