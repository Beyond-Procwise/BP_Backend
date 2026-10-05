"""Bucket edges, boundaries and the once-per-bucket / suppression lifecycle (pure)."""
from datetime import date

import pytest

from src.services.contract_expiry.buckets import (
    EXPIRED, NO_END_DATE, Desired, bucket_labels, classify, desired_alerts,
    is_active_demand, plan_changes,
)

AS_OF = date(2026, 10, 5)
M = [3, 6, 9, 12, 18]
CFG = {"bucket_months": M, "lifecycle_status": "active", "alert_expired": True,
       "flag_missing_end_date": True}


def test_labels_follow_the_edges():
    assert bucket_labels(M) == ["0-3", "3-6", "6-9", "9-12", "12-18"]
    assert bucket_labels([2, 4]) == ["0-2", "2-4"]


@pytest.mark.parametrize("end,expected", [
    (date(2026, 10, 4), EXPIRED),            # yesterday
    (date(2026, 10, 5), "0-3"),              # today is not expired
    (date(2027, 1, 5), "0-3"),               # exactly 3 months -> SOONER bucket
    (date(2027, 1, 6), "3-6"),               # one day over
    (date(2027, 4, 5), "3-6"),
    (date(2027, 4, 6), "6-9"),
    (date(2027, 7, 5), "6-9"),
    (date(2027, 7, 6), "9-12"),
    (date(2027, 10, 5), "9-12"),
    (date(2027, 10, 6), "12-18"),
    (date(2028, 4, 5), "12-18"),
    (date(2028, 4, 6), None),                # beyond 18 months: no alert
])
def test_every_boundary_and_one_day_either_side(end, expected):
    assert classify(end, AS_OF, M) == expected


def test_month_end_and_leap_year():
    # 31 Aug + 6 months has no 31 Feb: clamps to 28 Feb (2027), 29 Feb (2028)
    assert classify(date(2027, 2, 28), date(2026, 8, 31), M) == "3-6"
    assert classify(date(2027, 3, 1), date(2026, 8, 31), M) == "6-9"
    assert classify(date(2028, 2, 29), date(2027, 8, 31), M) == "3-6"
    assert classify(date(2028, 3, 1), date(2027, 8, 31), M) == "6-9"


def test_edges_must_be_positive():
    with pytest.raises(ValueError):
        classify(AS_OF, AS_OF, [0, 3])
    with pytest.raises(ValueError):
        classify(AS_OF, AS_OF, [])


def _c(cid, end, status="Active"):
    return {"contract_id": cid, "contract_end_date": end, "contract_lifecycle_status": status}


def test_desired_scope_and_missing_end_date():
    d = desired_alerts(
        [_c("a", date(2026, 12, 1)), _c("b", None), _c("c", date(2026, 1, 1)),
         _c("d", date(2031, 1, 1)), _c("e", date(2026, 12, 1), "Expired")],
        {}, AS_OF, CFG)
    assert {(x.contract_id, x.bucket) for x in d} == {("a", "0-3"), ("b", NO_END_DATE), ("c", EXPIRED)}


def test_config_can_switch_off_expired_and_missing():
    cfg = {**CFG, "alert_expired": False, "flag_missing_end_date": False}
    d = desired_alerts([_c("b", None), _c("c", date(2026, 1, 1))], {}, AS_OF, cfg)
    assert d == []


def test_active_demand_status():
    inactive = ["closed", "Cancelled", "draft"]
    assert is_active_demand("In approval", inactive)
    assert is_active_demand(None, inactive)
    assert not is_active_demand("CLOSED", inactive)
    assert not is_active_demand(" cancelled ", inactive)


def _row(aid, cid, bucket, end, status="open"):
    return {"alert_id": aid, "contract_id": cid, "bucket": bucket, "end_date": end, "status": status}


def test_fires_once_per_bucket():
    d = Desired("a", "0-3", date(2026, 12, 1), 57, None)
    first = plan_changes([d], [])
    assert first.insert == [d]
    again = plan_changes([d], [_row(1, "a", "0-3", date(2026, 12, 1))])
    assert again.insert == [] and again.touch == [1]


def test_moving_bucket_clears_old_and_fires_new():
    old = _row(1, "a", "3-6", date(2027, 2, 1))
    now = Desired("a", "0-3", date(2027, 2, 1), 60, None)
    p = plan_changes([now], [old])
    assert p.insert == [now] and p.clear == [1]


def test_amendment_extending_end_date_clears_and_refires():
    old = _row(1, "a", "0-3", date(2026, 12, 1))
    amended = Desired("a", "9-12", date(2027, 8, 1), 300, None)
    p = plan_changes([amended], [old])
    assert p.insert == [amended] and p.clear == [1]


def test_demand_appearing_suppresses_and_closing_reopens():
    d_sup = Desired("a", "0-3", date(2026, 12, 1), 57, "DM-7")
    p = plan_changes([d_sup], [_row(1, "a", "0-3", date(2026, 12, 1))])
    assert p.suppress == [(1, d_sup)]
    d_free = Desired("a", "0-3", date(2026, 12, 1), 57, None)
    p = plan_changes([d_free], [_row(1, "a", "0-3", date(2026, 12, 1), "suppressed")])
    assert p.reopen == [(1, d_free)]


def test_contract_gone_from_scope_is_cleared():
    p = plan_changes([], [_row(1, "a", "0-3", date(2026, 12, 1))])
    assert p.clear == [1]
