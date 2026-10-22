import datetime as dt
from decimal import Decimal

import pytest

from src.services import commodity_index as CI
from src.services.contract_tools import store

SERIES = ("LME-AL-CASH", "LME", "Aluminium cash settlement", "USD", "USD/tonne",
          dt.date(2026, 1, 1), dt.date(2026, 6, 1), 6)
POINTS = [(dt.date(2026, 1, 1), Decimal("2000"), "lme.com", "carol", dt.datetime(2026, 7, 1, 9)),
          (dt.date(2026, 3, 1), Decimal("2100"), "lme.com", "carol", dt.datetime(2026, 7, 1, 9)),
          (dt.date(2026, 6, 1), Decimal("2060"), "lme.com", "dave", dt.datetime(2026, 7, 2, 9))]


def _rows(series=(SERIES,), points=POINTS):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "GROUP BY series_code" in sql and "LIMIT 6" in sql:
            return list(series)
        if "GROUP BY series_code" in sql:
            return list(series)
        if "DISTINCT ON (period_date)" in sql:
            lo, hi = params[1], params[2]
            return [p for p in points if lo <= p[0] <= hi]
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def _down(sql, params):
    raise store.StoreUnavailable("db down")


# ---- validate / load ------------------------------------------------------------------------
def test_validate_sorts_points_and_keeps_values_exact():
    out = CI.validate_series(" LME-AL-CASH ", "LME", [{"period_date": "2026-03-01", "value": "2100.5"},
                                                      {"period_date": "2026-01-01", "value": 2000}], unit="USD/tonne")
    assert out["series_code"] == "LME-AL-CASH" and out["unit"] == "USD/tonne" and out["currency"] is None
    assert [p["period_date"] for p in out["points"]] == [dt.date(2026, 1, 1), dt.date(2026, 3, 1)]
    assert out["points"][1]["value"] == Decimal("2100.5")


@pytest.mark.parametrize("args, words", [
    (("", "LME", [{"period_date": "2026-01-01", "value": 1}]), "series_code is required"),
    (("X", "", [{"period_date": "2026-01-01", "value": 1}]), "publisher is required"),
    (("X", "LME", []), "points is required"),
    (("X", "LME", "2000"), "points is required"),
    (("X", "LME", [{"period_date": "Jan", "value": 1}]), "date like"),
    (("X", "LME", [{"period_date": "2026-01-01", "value": "high"}]), "must be a number"),
    (("X", "LME", [{"period_date": "2026-01-01", "value": "nan"}]), "finite"),
    (("X", "LME", [{"period_date": "2026-01-01", "value": 1}, {"period_date": "2026-01-01", "value": 2}]), "repeats"),
    (("X", "LME", [1]), "must be an object"),
])
def test_validate_refuses_each_bad_input(args, words):
    with pytest.raises(CI.InvalidSeries, match=words):
        CI.validate_series(*args)


def test_validate_caps_the_load():
    pts = [{"period_date": (dt.date(2000, 1, 1) + dt.timedelta(days=i)).isoformat(), "value": 1} for i in range(CI.MAX_POINTS_PER_LOAD + 1)]
    with pytest.raises(CI.InvalidSeries, match="at most"):
        CI.validate_series("X", "LME", pts)


def test_load_appends_one_row_per_point_signed_by_the_caller():
    writes = []
    out = CI.load_series("LME-AL-CASH", "LME", [{"period_date": "2026-03-01", "value": 2100},
                                                 {"period_date": "2026-01-01", "value": 2000}], "carol",
                         source_reference="lme.com", execute=lambda sql, p: writes.append((sql, p)))
    assert len(writes) == 2 and all("INSERT INTO proc.bp_commodity_index" in s for s, _ in writes)
    assert [p[5] for _, p in writes] == [dt.date(2026, 1, 1), dt.date(2026, 3, 1)] and writes[0][1][-1] == "carol"
    assert out == {"series_code": "LME-AL-CASH", "publisher": "LME", "points_loaded": 2, "first_period": "2026-01-01",
                   "last_period": "2026-03-01", "recorded_by": "carol"}


def test_load_needs_a_person_and_writes_nothing_on_a_bad_point():
    def boom(sql, p):
        raise AssertionError("must not write")
    with pytest.raises(CI.InvalidSeries, match="sign in"):
        CI.load_series("X", "LME", [{"period_date": "2026-01-01", "value": 1}], "", execute=boom)
    with pytest.raises(CI.InvalidSeries):
        CI.load_series("X", "LME", [{"period_date": "2026-01-01", "value": 1}, {"period_date": "x", "value": 1}],
                       "carol", execute=boom)


# ---- lookup --------------------------------------------------------------------------------------
def test_lookup_gives_the_series_its_points_and_the_move_between_first_and_last():
    out = CI.lookup("LME-AL-CASH", rows=_rows())
    assert out["found"] and out["series"]["publisher"] == "LME" and out["series"]["periods"] == 6
    assert out["points_in_window"] == 3 and out["points"][0] == {
        "period_date": "2026-01-01", "value": 2000.0, "source_reference": "lme.com", "recorded_by": "carol",
        "loaded_at": "2026-07-01T09:00:00"}
    assert out["move"]["from_value"] == 2000.0 and out["move"]["to_value"] == 2060.0 and out["move"]["change_pct"] == 3.0
    assert "not a forecast" in out["move"]["note"] and "nothing is interpolated" in out["note"]


def test_window_limits_the_points_and_the_move():
    rows = _rows()
    out = CI.lookup("LME-AL-CASH", "2026-02-01", "2026-06-30", rows=rows)
    assert out["points_in_window"] == 2 and out["move"]["from_period"] == "2026-03-01"
    assert out["move"]["change_pct"] == pytest.approx(-1.9048, abs=1e-4)
    assert rows.seen[-1][1] == ["LME-AL-CASH", dt.date(2026, 2, 1), dt.date(2026, 6, 30)]
    one = CI.lookup("LME-AL-CASH", "2026-03-01", "2026-03-01", rows=_rows())
    assert one["move"] is None and "no move can be given" in one["note"]
    none = CI.lookup("LME-AL-CASH", "2027-01-01", rows=_rows())
    assert none["found"] and none["points"] == [] and "not available" in none["note"]


def test_a_zero_first_value_has_no_percentage():
    pts = [(dt.date(2026, 1, 1), Decimal("0"), None, "c", None), (dt.date(2026, 2, 1), Decimal("5"), None, "c", None)]
    out = CI.lookup("LME-AL-CASH", rows=_rows(points=pts))
    assert out["move"]["change_pct"] is None and "undefined" in out["move"]["note"]


def test_bad_dates_and_order_are_refused():
    assert "date like" in CI.lookup("LME-AL-CASH", "soon", rows=_rows())["note"]
    assert CI.lookup("LME-AL-CASH", "2026-06-01", "2026-01-01", rows=_rows())["note"] == "start is after end."


def test_no_series_means_not_available_never_an_estimate():
    out = CI.lookup("copper", rows=_rows(series=()))
    assert out["found"] is False and out["loaded"] is False and "do not estimate" in out["note"]


def test_words_match_and_an_exact_code_wins_over_partial_matches():
    other = ("LME-AL-3M", "LME", "Aluminium 3 month", "USD", "USD/tonne", dt.date(2026, 1, 1), dt.date(2026, 6, 1), 6)
    out = CI.lookup("aluminium", rows=_rows(series=(SERIES, other)))
    assert out["found"] is False and out["ambiguous"] and [c["series_code"] for c in out["candidates"]] == ["LME-AL-CASH", "LME-AL-3M"]
    out = CI.lookup("LME-AL-CASH", rows=_rows(series=(SERIES, other)))
    assert out["found"] and out["series"]["series_code"] == "LME-AL-CASH"


def test_without_a_code_the_loaded_series_are_listed():
    out = CI.lookup(rows=_rows())
    assert out["found"] and out["series"][0]["series_code"] == "LME-AL-CASH" and "Give the series code" in out["note"]
    out = CI.lookup(rows=_rows(series=()))
    assert out["found"] is False and "not available" in out["note"]


def test_outage_is_not_no_series():
    out = CI.lookup("LME-AL-CASH", rows=_down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]
    assert CI.lookup(rows=_down)["available"] is False

    def later_down(sql, params):
        if "GROUP BY series_code" in sql:
            return [SERIES]
        raise store.StoreUnavailable("db down")
    assert CI.lookup("LME-AL-CASH", rows=later_down)["available"] is False


def test_long_series_show_both_ends():
    pts = [(dt.date(2000, 1, 1) + dt.timedelta(days=i), Decimal(i + 1), None, "c", None) for i in range(300)]
    out = CI.lookup("LME-AL-CASH", rows=_rows(points=pts))
    assert out["points_in_window"] == 300 and len(out["points"]) == CI.MAX_POINTS_RETURNED
    assert out["points"][-1]["period_date"] == pts[-1][0].isoformat() and "Showing the first and last" in out["note"]
    assert out["move"]["to_value"] == 300.0
