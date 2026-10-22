import json
from datetime import date, datetime
from decimal import Decimal

from src.services import supplier_tools as ST

TODAY = date(2026, 10, 21)


def _down(sql, params):
    raise RuntimeError("db")


def _drow(sid="7", name="Acme", pref=True, ctry="UK", ins=date(2027, 1, 1), po=2, inv=3, qt=1, iso=True):
    return (sid, name, pref, ctry, None, iso, False, False, ins, po, inv, qt)


def test_discovery_reports_evidence_and_says_it_is_not_a_market_scan():
    out = ST.discover("laptop", rows=lambda s, p: [_drow()], today=TODAY)
    s = out["suppliers"][0]
    assert s["evidence"] == {"po_lines": 2, "invoice_lines": 3, "quote_lines": 1}
    assert s["certifications_on_file"] == ["ISO 14001"] and s["insurance_status"] == "in date"
    assert "not a market scan" in out["note"]


def test_discovery_expired_or_missing_insurance_is_stated():
    out = ST.discover("laptop", rows=lambda s, p: [_drow(ins=datetime(2026, 1, 1)), _drow("8", ins=None)], today=TODAY)
    assert [s["insurance_status"] for s in out["suppliers"]] == ["expired", "not on file"]


def test_discovery_filters_reach_the_query_as_parameters_not_text():
    seen = {}

    def rows(sql, params):
        seen.update(sql=sql, params=params)
        return []
    ST.discover("lap_top%", country="UK", preferred_only=True, rows=rows)
    assert "UK" in seen["params"] and "%lap\\_top\\%%" in seen["params"] and "UK" not in seen["sql"]
    assert "is_preferred_supplier IS TRUE" in seen["sql"]


def test_discovery_nothing_found_does_not_claim_no_suppliers_exist():
    out = ST.discover("zeppelin", rows=lambda s, p: [], today=TODAY)
    assert out["found"] is False and "not covered" in out["note"]


def test_discovery_outage_and_short_item_are_not_empty_results():
    assert ST.discover("laptop", rows=_down)["available"] is False
    assert ST.discover("x")["found"] is False


def test_discovery_limit_is_capped_and_overflow_said():
    out = ST.discover("pen", limit=999, rows=lambda s, p: [_drow(str(i)) for i in range(ST.MAX_LIMIT + 1)], today=TODAY)
    assert len(out["suppliers"]) == ST.MAX_LIMIT and "narrow" in out["note"]


_L = ("7", "Acme Ltd", "Acme", "goods", "ltd", "UK", "123", True, None, 5000, True, False, True,
      "liability", 1000000, date(2025, 1, 1), "GBP", "DAP", 14, "UK", "acme.example", 12)


def test_lookup_profile_never_includes_bank_or_contacts():
    out = ST.lookup("acme", rows=lambda s, p: [_L], today=TODAY)
    p = out["suppliers"][0]
    assert p["risk_score"] == "not scored" and p["insurance"]["insurance_status"] == "expired"
    text = repr(p).lower() + ST._LOOKUP_SQL.lower()
    assert not any(w in text for w in ("iban", "swift", "bank_", "contact_"))


def test_lookup_ambiguous_name_asks_which_one():
    out = ST.lookup("acme", rows=lambda s, p: [_L, _L], today=TODAY)
    assert "confirm which" in out["note"]


def test_lookup_unknown_and_outage_differ():
    assert "ask the user" in ST.lookup("nobody", rows=lambda s, p: [])["note"]
    assert ST.lookup("acme", rows=_down)["available"] is False


def test_lookup_output_is_plain_json_even_with_decimal_amounts():
    row = _L[:9] + (Decimal("5000.00"),) + _L[10:14] + (Decimal("1000000.00"),) + _L[15:]
    json.dumps(ST.lookup("acme", rows=lambda s, p: [row], today=TODAY))


def test_lookup_matches_part_of_a_name_and_an_exact_id():
    seen = {}

    def rows(sql, params):
        seen["params"] = params
        return []
    ST.lookup("acme", rows=rows)
    assert seen["params"] == ["acme", "%acme%", "%acme%"]


def test_lookup_says_how_it_matched_and_a_lone_partial_match_must_be_confirmed():
    assert ST.lookup("7", rows=lambda s, p: [_L], today=TODAY)["suppliers"][0]["match"] == "exact id"
    full = ST.lookup("Acme Ltd", rows=lambda s, p: [_L], today=TODAY)
    assert full["suppliers"][0]["match"] == "exact name" and "confirm with the user" not in full["note"]
    part = ST.lookup("acm", rows=lambda s, p: [_L], today=TODAY)
    assert part["suppliers"][0]["match"] == "partial name" and "confirm with the user" in part["note"]
