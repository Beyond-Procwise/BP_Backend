import datetime as dt

from src.services.contract_tools import store, tracker

TODAY = dt.date(2026, 10, 22)
MASTER = ("C1", "Cloud MSA", "msa", "Acme", dt.date(2025, 1, 1), dt.date(2026, 12, 31), "active", "GBP",
          100000, "Yes", "12 months", "30 days")
CLAUSES = [
    (11, "D1", "C1", "Termination notice", "entitled_to", "27.4",
     "Either party may terminate this Agreement on ninety days' written notice."),
    (12, "D1", "C1", "Annual price review", "triggered_by", "9.2",
     "Prices shall be reviewed on each anniversary of the Commencement Date."),
    (13, "D1", "C1", "Delivery", "must_perform", "5.1", "The Supplier shall deliver the Goods to the Site."),
]
PARTIES = [(11, "Acme"), (11, "Buyer"), (12, "Acme"), (13, "Acme")]


def _rows(masters=(MASTER,), clauses=CLAUSES, parties=PARTIES):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_contract_master" in sql:
            return list(masters)
        if "bp_contract_obligation_party" in sql:
            return list(parties)
        if "bp_contract_obligation" in sql:
            return list(clauses)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def _down(sql, params=()):
    raise store.StoreUnavailable("db down")


def test_needs_a_contract():
    assert tracker.track(rows=_rows())["found"] is False


def test_calendar_comes_from_stored_fields_and_notice_deadline_is_never_computed():
    out = tracker.track("C1", rows=_rows(), today=TODAY)
    assert out["found"] and out["in_register"] and out["contract"]["end_date"] == "2026-12-31"
    events = {e["event"]: e for e in out["calendar"]}
    assert events["expiry"]["days_from_today"] == 70 and events["expiry"]["within_horizon"] is True
    assert events["auto-renewal"]["source"] == "auto_renew_flag" and events["auto-renewal"]["renewal_term"] == "12 months"
    assert "notice deadline" not in str([e.get("date") for e in out["calendar"]])
    assert not any(e["event"] == "notice deadline" for e in out["calendar"])
    assert out["registered"] is False and out["scheduled"] is False
    assert "no notice deadline is computed" in out["note"]


def test_only_administration_clauses_are_listed_and_notice_ones_are_flagged():
    out = tracker.track("C1", rows=_rows(), today=TODAY)
    refs = [c["clause_ref"] for c in out["administration_clauses"]]
    assert refs == ["9.2", "27.4"], refs  # 5.1 (plain delivery duty) is not an administration clause
    by_ref = {c["clause_ref"]: c for c in out["administration_clauses"]}
    assert by_ref["27.4"]["mentions_notice"] is True and by_ref["9.2"]["mentions_notice"] is False
    assert by_ref["27.4"]["quote"] == CLAUSES[0][6] and by_ref["27.4"]["parties"] == ["Acme", "Buyer"]
    assert out["notice_clauses"] == 1 and out["clauses_stored"] == 3


def test_no_auto_renew_means_no_renewal_event_and_expired_is_said():
    master = MASTER[:5] + (dt.date(2026, 9, 1),) + MASTER[6:9] + ("No",) + MASTER[10:]
    out = tracker.track("C1", rows=_rows(masters=(master,)), today=TODAY)
    assert [e["event"] for e in out["calendar"]] == ["expiry"]
    assert out["calendar"][0]["passed"] is True and out["calendar"][0]["days_from_today"] == -51


def test_contract_read_but_not_in_register_still_gives_its_clauses():
    out = tracker.track("D1", rows=_rows(masters=()), today=TODAY)
    assert out["found"] and out["in_register"] is False and out["contract"] is None and out["calendar"] == []
    assert len(out["administration_clauses"]) == 2 and "not in the register" in out["note"]


def test_in_register_but_never_read_says_clauses_may_still_exist():
    out = tracker.track("C1", rows=_rows(clauses=[], parties=[]), today=TODAY)
    assert out["found"] and out["clauses_read"] is False and out["administration_clauses"] == []
    assert "not been read" in out["note"]


def test_unknown_contract_asks_for_the_facts_unverified():
    out = tracker.track("Nothing", rows=_rows(masters=(), clauses=[], parties=[]), today=TODAY)
    assert out["found"] is False and "unverified" in out["note"]


def test_several_matches_ask_which():
    other = ("C2",) + MASTER[1:]
    out = tracker.track("Cloud", rows=_rows(masters=(MASTER, other)), today=TODAY)
    assert out["found"] is False and out["ambiguous"] is True
    assert [c["contract_id"] for c in out["candidates"]] == ["C1", "C2"]


def test_supplier_narrows_the_register_query():
    rows = _rows()
    tracker.track("Cloud", supplier="acme", rows=rows, today=TODAY)
    sql, params = rows.seen[0]
    assert "supplier_name, '') ILIKE" in sql and params == ["Cloud", "%Cloud%", "%acme%"]


def test_outage_is_not_reported_as_no_contract():
    out = tracker.track("C1", rows=_down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]

    def clauses_down(sql, params):
        if "bp_contract_master" in sql:
            return [MASTER]
        raise store.StoreUnavailable("db down")
    out = tracker.track("C1", rows=clauses_down, today=TODAY)
    assert out["found"] is False and out["available"] is False


def test_horizon_is_bounded_and_defaults():
    assert tracker.track("C1", horizon_days="soon", rows=_rows(), today=TODAY)["horizon_days"] == tracker.DEFAULT_HORIZON_DAYS
    assert tracker.track("C1", horizon_days=99999, rows=_rows(), today=TODAY)["horizon_days"] == tracker.MAX_HORIZON_DAYS
    out = tracker.track("C1", horizon_days=30, rows=_rows(), today=TODAY)
    assert out["calendar"][0]["within_horizon"] is False
