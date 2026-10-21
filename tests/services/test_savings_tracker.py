import datetime as dt

from src.services import savings_tracker as ST


def _row(oid, src, typ, gbp, day, sup=None):
    return {"outcome_id": oid, "source_type": "finding", "source_id": src, "outcome_type": typ,
            "amount": gbp, "currency": "GBP", "amount_gbp": gbp, "supersedes_id": sup,
            "valid_from": dt.date(2026, 10, day), "recorded_at": dt.datetime(2026, 10, day)}


def test_saved_counts_settled_money_and_keeps_claims_apart():
    rows = [_row(1, "a", "recovered", 100.0, 1), _row(2, "b", "claimed", 50.0, 2), _row(3, "c", "avoided", 25.0, 3)]
    out = ST.track(load=lambda: rows)["totals"]
    assert out["saved_gbp"] == 125.0 and out["claimed_open_gbp"] == 50.0


def test_a_correction_replaces_what_it_supersedes():
    rows = [_row(1, "a", "recovered", 100.0, 1), _row(2, "a", "recovered", 40.0, 1, sup=1)]
    assert ST.track(load=lambda: rows)["totals"]["saved_gbp"] == 40.0


def test_window_limits_settled_money_only():
    rows = [_row(1, "a", "recovered", 100.0, 1), _row(2, "b", "recovered", 10.0, 20)]
    assert ST.track(since="2026-10-10", load=lambda: rows)["totals"]["saved_gbp"] == 10.0


def test_unreadable_ledger_is_not_zero_savings():
    def boom():
        raise RuntimeError("db")
    out = ST.track(load=boom)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_empty_ledger_says_so_and_bad_date_is_refused():
    assert "no entries" in ST.track(load=lambda: [])["note"]
    assert ST.track(since="soon", load=lambda: [])["found"] is False
