import datetime as dt

from src.services.contract_tools import pipeline, store

TODAY = dt.date(2026, 10, 22)
# search row: contract_id, title, type, supplier, start, end, status, currency, value, auto_renew, renewal_term, governing_law, payment_terms
C1 = ("C1", "Cloud MSA", "msa", "Acme", dt.date(2025, 1, 1), dt.date(2026, 12, 31), "active", "GBP", 100000, "Yes", "12 months", "England", "30 days")
C2 = ("C2", "Cleaning", "services", "Borealis", dt.date(2024, 1, 1), dt.date(2026, 11, 15), "active", "GBP", 50000, "No", None, "England", "30 days")
C3 = ("C3", "Pallets", "supply", "Kestrel", dt.date(2025, 6, 1), dt.date(2027, 3, 1), "active", "EUR", 20000, "Yes", "12 months", None, None)
CLAUSES = {"C1": [(11, "D1", "C1", "Termination notice", "entitled_to", "27.4", "Either party may terminate on ninety days' written notice.")],
           "C2": [(21, "D2", "C2", "Delivery", "must_perform", "5.1", "The Supplier shall deliver the Goods.")],
           "C3": []}


def _rows(masters=(C1, C2, C3)):
    def rows(sql, params):
        if "bp_contract_master" in sql and "ORDER BY c.contract_end_date NULLS LAST LIMIT" in sql and "contract_id = %s OR c.contract_title" not in sql:
            return list(masters)
        if "bp_contract_master" in sql:
            cid = params[0]
            return [m for m in masters if m[0] == cid][:1]
        if "bp_contract_obligation_party" in sql:
            return []
        if "bp_contract_obligation" in sql:
            return list(CLAUSES.get(params[0], []))
        raise AssertionError(sql)
    return rows


def test_the_pipeline_lists_contracts_soonest_first_with_their_clauses_and_what_was_never_read():
    out = pipeline.build(180, rows=_rows(), today=TODAY)
    assert out["found"] and out["contracts_due"] == 3 and out["registered"] is False and out["scheduled"] is False
    assert [i["contract_id"] for i in out["detailed"]] == ["C2", "C1", "C3"]
    c2, c1, c3 = out["detailed"]
    assert c2["days_to_end"] == 24 and c2["clauses_read"] is True and c2["notice_clauses"] == [] and c2["decision_basis"] == "no notice clause stored"
    assert c1["notice_clauses"][0]["clause_ref"] == "27.4" and c1["decision_basis"].startswith("notice clause quoted")
    assert c1["calendar"][1]["event"] == "auto-renewal"
    assert c3["clauses_read"] is False and c3["decision_basis"] == "clauses never read for this contract"
    assert out["auto_renewing"] == ["C1", "C3"] and out["not_read_for_clauses"] == ["C3"] and out["no_notice_clause_stored"] == ["C2"]
    assert "not a stored field" in out["note"]


def test_detail_is_capped_and_the_rest_are_listed():
    out = pipeline.build(365, detail=1, rows=_rows(), today=TODAY)
    assert [i["contract_id"] for i in out["detailed"]] == ["C2"] and [c["contract_id"] for c in out["listed_only"]] == ["C1", "C3"]
    assert pipeline.build(detail=999, rows=_rows(), today=TODAY)["detailed"][0]["contract_id"] == "C2"


def test_horizon_defaults_and_bounds():
    assert pipeline.build("soon", rows=_rows(), today=TODAY)["horizon_days"] == pipeline.DEFAULT_HORIZON_DAYS
    assert pipeline.build(99999, rows=_rows(), today=TODAY)["horizon_days"] == pipeline.MAX_HORIZON_DAYS


def test_nothing_due_and_outage_are_told_apart():
    out = pipeline.build(30, rows=_rows(masters=()), today=TODAY)
    assert out["found"] is False and "No contract on the register ends" in out["note"]

    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = pipeline.build(30, rows=down, today=TODAY)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_the_summary_comes_first_and_counts_the_pipeline():
    out = pipeline.build(180, rows=_rows(), today=TODAY)
    assert list(out)[:2] == ["found", "summary"]
    assert out["summary"] == ("3 contracts on the register end within 180 days; 2 auto-renew; of the 3 detailed, 1 were never "
                              "read for clauses and 1 have no notice clause stored, so their decision dates are unknown.")
