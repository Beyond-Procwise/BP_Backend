"""The brief's expiry signal reads the rule's buckets; it does not restate them."""
from datetime import date

from src.services.brief_signals_service import MAX_ITEMS, expiry_buckets_signal
from src.services.contract_expiry.buckets import Desired

TODAY = date(2026, 10, 5)
RATES = {"GBP": 0.8, "USD": 1.0}


def _contract(cid, value=100.0, ccy="GBP", title=None, supplier=None):
    return {"contract_id": cid, "contract_title": title or f"T-{cid}", "supplier_name": supplier,
            "total_contract_value": value, "currency": ccy}


def _ev(items, months=(1, 6, 12, 24)):
    """items: (contract_id, bucket, end, suppressed_by, contract kwargs)"""
    contracts, desired = {}, []
    for cid, bucket, end, sup, kw in items:
        contracts[cid] = _contract(cid, **kw)
        days = (end - TODAY).days if end else None
        desired.append(Desired(cid, bucket, end, days, sup))
    return {"cfg": {"bucket_months": list(months)}, "contracts": contracts, "desired": desired}


def test_absent_when_nothing_to_alert():
    assert expiry_buckets_signal(_ev([]), TODAY, RATES) is None


def test_the_ladder_lists_every_bucket_in_rule_order_even_empty_ones():
    ev = _ev([("a", "6-12", date(2027, 3, 1), None, {})])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert [b["bucket"] for b in sig["buckets"]] == ["0-1", "1-6", "6-12", "12-24"]
    assert [b["count"] for b in sig["buckets"]] == [0, 0, 1, 0]


def test_the_edges_come_from_the_rule_not_the_code():
    ev = _ev([("a", "0-2", date(2026, 11, 20), None, {})], months=(2, 4))
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert [b["bucket"] for b in sig["buckets"]] == ["0-2", "2-4"]
    assert sig["headlineBucket"] == "0-2"


def test_headline_is_the_soonest_bucket_with_anything_in_it():
    ev = _ev([("a", "6-12", date(2027, 3, 1), None, {}), ("b", "12-24", date(2028, 1, 1), None, {})])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert sig["headlineBucket"] == "6-12" and sig["count"] == 1
    assert sig["withinDays"] == (date(2027, 10, 5) - TODAY).days     # 12 months out
    assert sig["nearest"]["days"] == (date(2027, 3, 1) - TODAY).days


def test_expired_and_missing_end_date_are_counted_apart_never_in_a_bucket():
    ev = _ev([("a", "0-1", date(2026, 10, 9), None, {}),
              ("x", "EXPIRED", date(2026, 1, 1), None, {}),
              ("y", "EXPIRED", date(2025, 1, 1), None, {}),
              ("n", "NO_END_DATE", None, None, {})])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert sig["count"] == 1
    assert sig["expired"]["count"] == 2 and sig["noEndDate"]["count"] == 1
    assert sum(b["count"] for b in sig["buckets"]) == 1


def test_a_contract_in_an_active_demand_is_left_out_but_counted_as_suppressed():
    ev = _ev([("a", "0-1", date(2026, 10, 9), "DM-7", {}), ("b", "0-1", date(2026, 10, 10), None, {})])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert sig["count"] == 1 and sig["suppressed"] == 1
    assert [i["name"] for i in sig["items"]] == ["T-b"]


def test_nothing_forward_means_no_headline_but_the_expired_still_show():
    sig = expiry_buckets_signal(_ev([("x", "EXPIRED", date(2026, 1, 1), None, {})]), TODAY, RATES)
    assert sig["count"] == 0 and "items" not in sig and sig["expired"]["count"] == 1


def test_value_totals_and_partial_flag():
    ev = _ev([("a", "0-1", date(2026, 10, 9), None, {"value": 800.0}),
              ("b", "0-1", date(2026, 10, 10), None, {"value": None})])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert sig["count"] == 2 and sig["valueGbp"] == 800.0 and sig["valuePartial"] is True


def test_items_are_a_sample_but_the_count_is_whole():
    ev = _ev([(f"c{i}", "0-1", date(2026, 10, 6 + i % 20), None, {}) for i in range(MAX_ITEMS + 5)])
    sig = expiry_buckets_signal(ev, TODAY, RATES)
    assert sig["count"] == MAX_ITEMS + 5 and len(sig["items"]) == MAX_ITEMS
    assert [i["days"] for i in sig["items"]] == sorted(i["days"] for i in sig["items"])
