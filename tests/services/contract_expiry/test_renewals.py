"""The renewals list: same evaluation as the brief, one row per alert."""
from datetime import date

from src.services.contract_expiry.buckets import Desired
from src.services.contract_expiry.renewals import renewals_payload

TODAY = date(2026, 10, 5)
RATES = {"GBP": 0.8, "USD": 1.0}


def _ev(items):
    contracts, desired = {}, []
    for cid, bucket, end, sup, extra in items:
        contracts[cid] = {"contract_id": cid, "contract_title": f"T-{cid}", "supplier_name": None,
                          "total_contract_value": 800.0, "currency": "GBP", "auto_renew_flag": "Yes",
                          "renewal_term": "12 months", "spend_category": "IT", **extra}
        desired.append(Desired(cid, bucket, end, (end - TODAY).days if end else None, sup))
    return {"cfg": {"bucket_months": [1, 6, 12, 24]}, "contracts": contracts, "desired": desired}


def test_one_row_per_alert_with_the_fields_the_screen_shows():
    out = renewals_payload(_ev([("a", "0-1", date(2026, 10, 9), None, {})]), TODAY, RATES)
    row = out["items"][0]
    assert out["total"] == 1 and out["asOf"] == "2026-10-05"
    assert row["contractId"] == "a" and row["endDate"] == "2026-10-09" and row["daysToEnd"] == 4
    assert row["bucket"] == "0-1" and row["state"] == "open" and row["autoRenew"] is True
    assert row["value"] == 800.0 and row["valueGbp"] == 800.0 and row["renewalTerm"] == "12 months"


def test_contracts_in_an_active_demand_are_listed_but_marked():
    out = renewals_payload(_ev([("a", "0-1", date(2026, 10, 9), "DM-7", {})]), TODAY, RATES)
    assert out["items"][0]["state"] == "inDemand" and out["items"][0]["demandId"] == "DM-7"
    assert out["summary"] is None            # nothing is OPEN, so the brief's card is absent


def test_order_is_soonest_first_then_expired_most_recent_first_then_no_end_date():
    ev = _ev([("n", "NO_END_DATE", None, None, {}),
              ("old", "EXPIRED", date(2024, 1, 1), None, {}),
              ("recent", "EXPIRED", date(2026, 9, 1), None, {}),
              ("late", "6-12", date(2027, 5, 1), None, {}),
              ("soon", "0-1", date(2026, 10, 9), None, {})])
    ids = [i["contractId"] for i in renewals_payload(ev, TODAY, RATES)["items"]]
    assert ids == ["soon", "late", "recent", "old", "n"]


def test_a_missing_value_stays_missing_it_is_not_zero():
    ev = _ev([("a", "0-1", date(2026, 10, 9), None, {"total_contract_value": None})])
    row = renewals_payload(ev, TODAY, RATES)["items"][0]
    assert row["value"] is None and row["valueGbp"] is None


def test_the_summary_is_the_briefs_own_signal():
    ev = _ev([("a", "0-1", date(2026, 10, 9), None, {}), ("x", "EXPIRED", date(2026, 1, 1), None, {})])
    s = renewals_payload(ev, TODAY, RATES)["summary"]
    assert s["count"] == 1 and s["expired"]["count"] == 1


def test_bucket_names_on_the_wire_are_plain_words_the_output_filter_leaves_alone():
    ev = _ev([("x", "EXPIRED", date(2026, 1, 1), None, {}), ("n", "NO_END_DATE", None, None, {}),
              ("a", "0-1", date(2026, 10, 9), None, {})])
    buckets = {i["contractId"]: i["bucket"] for i in renewals_payload(ev, TODAY, RATES)["items"]}
    assert buckets == {"x": "expired", "n": "noEndDate", "a": "0-1"}


def test_the_output_filter_leaves_the_whole_payload_untouched():
    import json
    from src.services import output_safety as osafe
    ev = _ev([("x", "EXPIRED", date(2026, 1, 1), None, {}), ("n", "NO_END_DATE", None, None, {}),
              ("a", "0-1", date(2026, 10, 9), "DM-7", {})])
    payload = json.loads(json.dumps(renewals_payload(ev, TODAY, RATES), default=str))
    assert osafe.scrub_payload(json.loads(json.dumps(payload)), where="/spendiq/contract-renewals") == payload


def test_active_contracts_reads_live_rows_only_and_carries_no_supplier_name(monkeypatch):
    from datetime import date as _d
    from src.services.contract_expiry import renewals

    seen = {}

    class Cur:
        description = [("contract_id",), ("contract_title",), ("spend_category",), ("contract_end_date",),
                       ("total_contract_value",), ("currency",), ("auto_renew_flag",), ("cost_centre_id",)]
        def execute(self, sql, params): seen["sql"], seen["params"] = sql, params
        def fetchall(self):
            from decimal import Decimal
            return [("C1", "Hosting", "IT", _d(2026, 12, 1), Decimal("100.50"), "GBP", "Yes", "CC-1")]

    class Conn:
        def cursor(self): return Cur()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(renewals, "get_conn", lambda: Conn())
    rows = renewals.active_contracts(_d(2026, 10, 5))
    assert rows == [{"contract_id": "C1", "contract_title": "Hosting", "spend_category": "IT",
                     "contract_end_date": "2026-12-01", "total_contract_value": 100.5,
                     "currency": "GBP", "auto_renew_flag": "Yes", "cost_centre_id": "CC-1"}]
    assert seen["params"] == (_d(2026, 10, 5),) and ">= %s" in seen["sql"]
    assert "supplier" not in seen["sql"].lower()
