"""OTIF: on time AND in full, measured only on deliveries we can actually assess."""
import pytest

from src.services.supplier_performance.otif import otif

TOL = dict(late_tolerance_days=0, in_full_tolerance=0.0)


def d(promised="2026-03-10", received="2026-03-10", ordered=100.0, got=100.0):
    return {"promised_date": promised, "received_date": received,
            "ordered_qty": ordered, "received_qty": got}


def test_on_time_and_in_full_counts_as_otif():
    r = otif(deliveries=[d()], **TOL)
    assert r["otif_rate"] == 1.0 and r["on_time_rate"] == 1.0 and r["in_full_rate"] == 1.0


def test_late_is_not_on_time_and_short_is_not_in_full():
    r = otif(deliveries=[d(), d(received="2026-03-12"), d(got=90.0), d(received="2026-03-12", got=90.0)],
             **TOL)
    assert r["on_time_rate"] == pytest.approx(0.5)
    assert r["in_full_rate"] == pytest.approx(0.5)
    assert r["otif_rate"] == pytest.approx(0.25)
    assert r["otif_score"] == pytest.approx(25.0)


def test_late_tolerance_and_early_delivery():
    ok = d(received="2026-03-12")
    assert otif(deliveries=[ok], late_tolerance_days=2, in_full_tolerance=0.0)["otif_rate"] == 1.0
    early = d(received="2026-03-01")
    assert otif(deliveries=[early], **TOL)["otif_rate"] == 1.0          # early is not penalised...
    strict = otif(deliveries=[early], early_tolerance_days=2, **TOL)    # ...unless a limit is set
    assert strict["on_time_rate"] == 0.0


def test_in_full_tolerance_and_over_delivery():
    assert otif(deliveries=[d(got=95.0)], late_tolerance_days=0,
                in_full_tolerance=0.05)["in_full_rate"] == 1.0
    assert otif(deliveries=[d(got=95.0)], **TOL)["in_full_rate"] == 0.0
    over = otif(deliveries=[d(got=120.0)], **TOL)
    assert over["in_full_rate"] == 1.0 and over["over_delivered"] == 1


def test_a_delivery_we_cannot_assess_is_counted_not_failed():
    r = otif(deliveries=[d(), d(promised=None), d(got=None), d(received=None, got=None)], **TOL)
    assert r["deliveries"] == 4
    assert r["otif_assessed"] == 1
    assert r["otif_rate"] == 1.0
    assert r["on_time_assessed"] == 2        # the one with no promised date and the one with no receipt drop out
    assert r["in_full_assessed"] == 2


def test_each_rate_uses_its_own_assessable_deliveries():
    r = otif(deliveries=[d(promised=None, got=90.0), d()], **TOL)
    assert r["in_full_rate"] == pytest.approx(0.5)     # both have quantities
    assert r["on_time_rate"] == 1.0                    # only one has dates
    assert r["otif_rate"] == 1.0                       # only one is assessable for both


def test_nothing_assessable_is_unassessed_not_zero():
    assert otif(deliveries=[d(promised=None, received=None, ordered=None, got=None)], **TOL) is None
    assert otif(deliveries=[], **TOL) is None


def test_only_dates_still_gives_on_time_but_no_otif():
    r = otif(deliveries=[d(ordered=None, got=None)], **TOL)
    assert r["on_time_rate"] == 1.0
    assert r["in_full_rate"] is None and r["otif_rate"] is None and r["otif_score"] is None


def test_unreadable_dates_are_unassessed_not_late():
    r = otif(deliveries=[d(received="not a date"), d()], **TOL)
    assert r["on_time_assessed"] == 1


def test_a_received_date_before_nothing_is_a_future_receipt_is_still_assessed():
    # a receipt dated the same day as promised is on time; the formula does not look at "today"
    assert otif(deliveries=[d(promised="2026-03-10", received="2026-03-10")], **TOL)["on_time_rate"] == 1.0


def test_negative_tolerances_are_refused():
    with pytest.raises(ValueError):
        otif(deliveries=[d()], late_tolerance_days=-1, in_full_tolerance=0.0)
    with pytest.raises(ValueError):
        otif(deliveries=[d()], late_tolerance_days=0, in_full_tolerance=1.5)
