import datetime as dt

from src.services.contract_tools import clauses, redline, search, store


def _down(sql, params=()):
    raise store.StoreUnavailable("db down")


# ---- redline -------------------------------------------------------------------------------
def test_redline_reports_a_replaced_number():
    out = redline.compare("Payment within 30 days of invoice.", "Payment within 60 days of invoice.")
    assert out["compared"] and not out["identical"]
    assert out["changes"] == [{"change": "replaced", "original": "30", "revised": "60", "context": "Payment within"}]


def test_redline_reports_added_and_removed_words():
    out = redline.compare("The Supplier shall deliver.", "The Supplier shall promptly deliver.")
    assert [(c["change"], c["revised"]) for c in out["changes"]] == [("added", "promptly")]
    out = redline.compare("The Supplier shall promptly deliver.", "The Supplier shall deliver.")
    assert [(c["change"], c["original"]) for c in out["changes"]] == [("removed", "promptly")]


def test_redline_identical_texts_say_identical_not_empty():
    out = redline.compare("A b c", "A  b\nc")
    assert out["identical"] is True and out["changes"] == []


def test_redline_refuses_missing_empty_or_huge_input():
    assert redline.compare(None, "x")["compared"] is False
    assert redline.compare("   ", "x")["compared"] is False
    assert redline.compare("x" * (redline.MAX_CHARS + 1), "x")["compared"] is False


# ---- search --------------------------------------------------------------------------------
def test_search_needs_a_criterion():
    assert search.search(rows=lambda s, p: [])["found"] is False


def test_search_outage_is_not_reported_as_no_contracts():
    out = search.search(query="cloud", rows=_down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_search_maps_rows_and_dates():
    row = ("C1", "Cloud MSA", "msa", "Acme", dt.date(2025, 1, 1), dt.date(2026, 12, 31), "active", "GBP",
           100000, True, "12 months", "England", "30 days")
    seen = {}

    def rows(sql, params):
        seen["sql"], seen["params"] = sql, params
        return [row]
    out = search.search(supplier="acme", rows=rows)
    assert out["contracts"][0]["end_date"] == "2026-12-31" and out["contracts"][0]["auto_renew"] is True
    assert "%acme%" in seen["params"] and "LIMIT" in seen["sql"]


def test_search_limit_is_capped_and_overflow_is_said():
    row = ("C", "t", "x", "s", None, None, "a", "GBP", 1, False, None, None, None)
    out = search.search(query="x", limit=500, rows=lambda s, p: [row] * (search.MAX_LIMIT + 1))
    assert out["count"] == search.MAX_LIMIT and "narrow" in out["note"]


def test_search_bad_window_is_refused():
    assert search.search(expires_within_days="soon", rows=lambda s, p: [])["found"] is False


# ---- clauses -------------------------------------------------------------------------------
_OB = [(1, "1520", None, "Payment", "must_perform", "5.1", "Payment shall be made within 30 days."),
       (2, "1520", None, "Audit", "entitled_to", "12", "The Authority may audit the Contractor."),
       (3, "1520", None, "Delivery", "must_perform", "3", "Delivery shall be completed by 31 March 2026.")]


def _rows(sql, params):
    return [(1, "Acme"), (1, "Authority")] if "entity_name" in sql else _OB


def test_clauses_are_returned_in_clause_order_with_parties():
    out = clauses.extract("1520", rows=_rows)
    assert [c["clause_ref"] for c in out["clauses"]] == ["3", "5.1", "12"]
    assert out["clauses"][1]["parties"] == ["Acme", "Authority"]


def test_clauses_filter_by_type_and_text():
    assert [c["clause_ref"] for c in clauses.extract("1520", clause_type="entitled_to", rows=_rows)["clauses"]] == ["12"]
    assert [c["clause_ref"] for c in clauses.extract("1520", containing="audit", rows=_rows)["clauses"]] == ["12"]


def test_clauses_unknown_contract_says_not_read_rather_than_none():
    out = clauses.extract("nope", rows=lambda s, p: [])
    assert out["found"] is False and out["read"] is False and "unchecked" in out["note"]


def test_clauses_outage_is_not_reported_as_no_clauses():
    out = clauses.extract("1520", rows=_down)
    assert out["available"] is False and "does not mean" in out["note"]


def test_clauses_refuse_a_missing_id_or_unknown_type():
    assert clauses.extract(None, rows=_rows)["found"] is False
    assert "must be one of" in clauses.extract("1", clause_type="payment", rows=_rows)["note"]


def test_filter_that_matches_nothing_still_says_clauses_exist():
    out = clauses.extract("1520", containing="zzz", rows=_rows)
    assert out["found"] is False and out["read"] is True and out["clauses_stored"] == 3
