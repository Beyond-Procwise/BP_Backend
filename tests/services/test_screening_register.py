import datetime as dt
import json
from decimal import Decimal

import pytest

from src.services import screening_register as SR
from src.services.contract_tools import store

TODAY = dt.date(2026, 10, 22)
SUPPLIERS = [("S1", "Acme Ltd")]


def _entry(**kw):
    base = {"supplier_id": "S1", "check_type": "sanctions", "result": "clear", "provider": "Dow Jones",
            "checked_on": "2026-10-01"}
    base.update(kw)
    return base


def _row(subject="supplier", result="clear", checked_on=dt.date(2026, 10, 1), valid_until=None, rating=None,
         score=None, details=None, provider="Dow Jones", reference="DJ-1", by="carol"):
    return (subject, result, rating, score, details, provider, reference, checked_on, valid_until, by,
            dt.datetime(2026, 10, 1, 9, 0))


def _rows(latest=(), signals=(), suppliers=SUPPLIERS):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_supplier s" in sql:
            return list(suppliers)
        if "DISTINCT ON (subject)" in sql:
            return list(latest)
        if "DISTINCT ON (check_type, subject)" in sql:
            return []
        if "supplier_risk_signals" in sql:
            return list(signals)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


# ---- validate / record ------------------------------------------------------------------------
def test_validate_accepts_a_complete_entry_and_fills_defaults():
    out = SR.validate(_entry(score="0.03", details=None), today=TODAY)
    assert out["subject"] == "supplier" and out["score"] == 0.03 and out["details"] == {}
    assert out["checked_on"] == dt.date(2026, 10, 1) and out["valid_until"] is None


@pytest.mark.parametrize("bad, words", [
    ({"supplier_id": ""}, "supplier_id"),
    ({"check_type": "credit"}, "check_type must be one of"),
    ({"result": "ok"}, "result must be one of"),
    ({"provider": " "}, "provider is required"),
    ({"checked_on": None}, "checked_on is required"),
    ({"checked_on": "soon"}, "date like"),
    ({"checked_on": "2026-10-23"}, "future"),
    ({"valid_until": "2026-09-01"}, "before checked_on"),
    ({"score": "high"}, "score must be a number"),
    ({"details": "x"}, "details must be an object"),
])
def test_validate_refuses_each_bad_field(bad, words):
    with pytest.raises(SR.InvalidEntry, match=words):
        SR.validate(_entry(**bad), today=TODAY)


def test_record_appends_with_the_signed_in_person_never_the_entry():
    seen = {}

    def execute(sql, params):
        seen["sql"], seen["params"] = sql, params
        return [(7, dt.datetime(2026, 10, 22, 10, 0))]
    out = SR.record(_entry(recorded_by="mallory", rating="Clear", reference="DJ-9"), "carol", execute=execute, today=TODAY)
    assert "INSERT INTO proc.bp_supplier_screening" in seen["sql"] and seen["params"][-1] == "carol"
    assert json.loads(seen["params"][6]) == {} and seen["params"][9] == dt.date(2026, 10, 1)
    assert out["screening_id"] == 7 and out["recorded_by"] == "carol" and out["checked_on"] == "2026-10-01"


def test_record_needs_a_person_and_refuses_before_writing():
    def boom(sql, params):
        raise AssertionError("must not write")
    with pytest.raises(SR.InvalidEntry, match="sign in"):
        SR.record(_entry(), "", execute=boom, today=TODAY)
    with pytest.raises(SR.InvalidEntry):
        SR.record(_entry(result="maybe"), "carol", execute=boom, today=TODAY)


# ---- lookup -------------------------------------------------------------------------------------
def test_no_record_is_not_performed_and_not_clear():
    out = SR.lookup("acme", "sanctions", rows=_rows(), today=TODAY)
    assert out["found"] and out["performed"] is False and out["checks"] == [] and out["flagged"] is False
    assert "NOT a clear result" in out["note"] and out["performed_by_this_tool"] is False


def test_latest_per_subject_with_provider_and_dates_and_expiry():
    latest = [_row(), _row(subject="director: J Smith", result="flagged", checked_on=dt.date(2026, 9, 1),
                           valid_until=dt.date(2026, 10, 1))]
    out = SR.lookup("S1", "sanctions", rows=_rows(latest=latest), today=TODAY)
    assert out["performed"] and out["flagged"] is True and out["expired"] is True
    by = {c["subject"]: c for c in out["checks"]}
    assert by["supplier"]["provider"] == "Dow Jones" and by["supplier"]["checked_on"] == "2026-10-01"
    assert by["supplier"]["expired"] is False and by["director: J Smith"]["expired"] is True
    assert "performed no check itself" in out["note"]


def test_a_not_performed_record_alone_counts_as_not_performed():
    out = SR.lookup("S1", "financial_health", rows=_rows(latest=[_row(result="not_performed")]), today=TODAY)
    assert out["performed"] is False and len(out["checks"]) == 1 and "NOT a clear result" in out["note"]


def test_rated_check_carries_rating_score_and_details_as_stated():
    latest = [_row(result="rated", rating="BBB", score=Decimal("0.021"), details='{"pd_12m": 0.021}', provider="D&B")]
    out = SR.lookup("S1", "financial_health", rows=_rows(latest=latest), today=TODAY)
    c = out["checks"][0]
    assert c["rating"] == "BBB" and c["score"] == 0.021 and c["details"] == {"pd_12m": 0.021} and c["provider"] == "D&B"


def test_risk_events_include_the_platforms_own_signals():
    signals = [("financial", Decimal("0.9"), "achilles", {"region": "EMEA"}, dt.datetime(2026, 10, 4, 9, 23))]
    out = SR.lookup("S1", "risk_event", rows=_rows(signals=signals), today=TODAY)
    assert out["performed"] is False and out["signals"][0]["source"] == "achilles"
    assert out["signals"][0]["severity"] == 0.9 and out["signals"][0]["occurred_at"].startswith("2026-10-04")
    out = SR.lookup("S1", "risk_event", rows=_rows(), today=TODAY)
    assert out["signals"] == [] and "No risk signals are recorded" in out["note"]


def test_other_types_do_not_read_signals():
    rows = _rows()
    SR.lookup("S1", "esg_rating", rows=rows, today=TODAY)
    assert not any("supplier_risk_signals" in sql for sql, _ in rows.seen)


def test_unknown_type_supplier_ambiguity_and_outage():
    assert SR.lookup("S1", "credit", rows=_rows())["found"] is False
    assert SR.lookup("", "sanctions", rows=_rows())["found"] is False
    out = SR.lookup("nobody", "sanctions", rows=_rows(suppliers=[]))
    assert out["found"] is False and "not on record" in out["note"]
    out = SR.lookup("acme", "sanctions", rows=_rows(suppliers=[("S1", "Acme Ltd"), ("S2", "Acme Logistics")]))
    assert out["ambiguous"] and [c["supplier_id"] for c in out["candidates"]] == ["S1", "S2"]

    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = SR.lookup("S1", "sanctions", rows=down)
    assert out["found"] is False and out["available"] is False and "not a clear result" in out["note"]

    def later_down(sql, params):
        if "bp_supplier s" in sql:
            return SUPPLIERS
        raise store.StoreUnavailable("db down")
    assert SR.lookup("S1", "sanctions", rows=later_down)["available"] is False


def test_history_lists_every_check_newest_first():
    rows_out = [(9, "sanctions") + _row(checked_on=dt.date(2026, 10, 1)),
                (3, "esg_rating") + _row(result="rated", rating="Silver", checked_on=dt.date(2026, 5, 1),
                                         valid_until=dt.date(2026, 6, 1))]

    def rows(sql, params):
        assert "ORDER BY checked_on DESC" in sql and params == ["S1", 100]
        return rows_out
    out = SR.history("S1", rows=rows, today=TODAY)
    assert [(c["screening_id"], c["check_type"], c["expired"]) for c in out] == [(9, "sanctions", False), (3, "esg_rating", True)]


def test_every_check_type_has_a_tool_name():
    assert {v["tool"] for v in SR.CHECK_TYPES.values()} == {
        "screen_sanctions", "financial_health_score", "esg_rating_lookup", "forced_labour_screen",
        "conflict_minerals_check", "monitor_risk_events"}


def test_summary_reads_every_check_type_in_one_call_and_lists_what_is_missing():
    latest = [("sanctions", "supplier", "clear", None, None, None, "Dow Jones", "DJ-1", dt.date(2026, 10, 1), None, "carol", dt.datetime(2026, 10, 1, 9)),
              ("sanctions", "director: J Smith", "flagged", None, None, None, "Dow Jones", "DJ-2", dt.date(2026, 9, 1), dt.date(2026, 10, 1), "carol", dt.datetime(2026, 9, 1, 9)),
              ("financial_health", "supplier", "rated", "BBB", Decimal("0.02"), '{"pd_12m": 0.02}', "D&B", None, dt.date(2026, 6, 1), dt.date(2026, 12, 1), "carol", dt.datetime(2026, 6, 1, 9))]
    seen = []

    def rows(sql, params):
        seen.append(sql)
        if "bp_supplier s" in sql:
            return SUPPLIERS
        if "DISTINCT ON (check_type, subject)" in sql:
            return latest
        if "supplier_risk_signals" in sql:
            return []
        raise AssertionError(sql)
    out = SR.summary("S1", rows=rows, today=TODAY)
    assert out["found"] and out["performed_by_this_tool"] is False and len(seen) == 3
    assert out["checks"]["sanctions"]["performed"] and out["checks"]["sanctions"]["flagged"] and out["checks"]["sanctions"]["expired"]
    assert out["checks"]["sanctions"]["subjects"] == ["supplier", "director: J Smith"]
    assert out["checks"]["financial_health"]["checks"][0]["rating"] == "BBB"
    assert out["not_performed"] == ["esg_rating", "forced_labour", "conflict_minerals", "risk_event"]
    assert out["flagged"] == ["sanctions"] and out["expired"] == ["sanctions"]
    assert "not clear" in out["note"] and "No risk signals" in out["note"]


def test_summary_with_nothing_recorded_is_not_clear_and_outage_is_told_apart():
    out = SR.summary("acme", rows=_rows(), today=TODAY)
    assert out["found"] and len(out["not_performed"]) == 6 and "NOT a clear result" in out["note"]

    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = SR.summary("acme", rows=down)
    assert out["found"] is False and out["available"] is False and "not a clear result" in out["note"]
    out = SR.summary("acme", rows=_rows(suppliers=[("S1", "Acme Ltd"), ("S2", "Acme Logistics")]))
    assert out["ambiguous"]


def test_the_summary_comes_first_and_names_what_is_missing_flagged_and_expired():
    out = SR.summary("acme", rows=_rows(), today=TODAY)
    assert list(out)[:2] == ["found", "summary"]
    assert out["summary"].startswith("Acme Ltd: 0 of 6 checks recorded; not performed: sanctions, financial_health")
    assert out["summary"].endswith("0 risk signals on the platform's feed. A check not performed is not clear.")
