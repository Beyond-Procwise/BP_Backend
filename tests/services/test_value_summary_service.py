"""Unit tests for the Value Found aggregation core. Pure functions on dicts —
no DB. Row shapes mirror proc.bp_extraction_discrepancy / proc.bp_opportunity."""
from datetime import datetime, timezone, timedelta

from src.services.value_summary_service import (
    parse_amount, discrepancy_delta, classify_discrepancy,
    classify_opportunity, dedupe, summarise,
)


def _disc(**kw):
    row = {
        "discrepancy_id": 1, "issue_type": "amount_over_po", "status": "open",
        "raw_value": "10950.00", "expected_value": "10000.00", "computed_value": "+950.00",
        "resolution_outcome": None, "recovered_amount": None, "query_sent_at": None,
        "doc_type": "invoice", "doc_pk_candidate": "INV-1042", "deal_id": "D-1",
        "supplier_name": "Techworld", "currency": "GBP", "notes": "",
        "created_at": datetime.now(timezone.utc) - timedelta(days=45),
    }
    row.update(kw)
    return row


def test_parse_amount_strict():
    assert parse_amount("£1,234.50") == 1234.5
    assert parse_amount("+950.00") == 950.0
    assert parse_amount("a" * 64) is None          # SHA-256-like → None, never a number
    assert parse_amount("1042 refund") is None     # partial numbers don't count


def test_delta_prefers_signed_computed_value():
    assert discrepancy_delta(_disc()) == 950.0
    # convention 2: no computed_value → observed − expected
    assert discrepancy_delta(_disc(computed_value=None)) == 950.0
    assert discrepancy_delta(_disc(computed_value="garbage", raw_value="x")) is None


def test_discrepancy_tiering():
    f = classify_discrepancy(_disc())
    assert f["tier"] == "verified" and f["amount_gbp"] == 950.0 and f["queryable"] is True
    assert f["age_days"] == 45
    # ignored (= dismissed false positive) and superseded never count
    assert classify_discrepancy(_disc(status="ignored")) is None
    assert classify_discrepancy(_disc(status="superseded")) is None
    # historical resolved with NULL outcome: stays in found, NOT recovered
    f = classify_discrepancy(_disc(status="resolved"))
    assert f["tier"] == "verified" and f["status"] == "resolved"
    # resolved as recovered: amount falls back to delta when recovered_amount is null
    f = classify_discrepancy(_disc(status="resolved", resolution_outcome="recovered"))
    assert f["tier"] == "verified" and f["recovered_gbp"] == 950.0
    f = classify_discrepancy(_disc(status="resolved", resolution_outcome="recovered",
                                   recovered_amount=500))
    assert f["recovered_gbp"] == 500.0
    # non-value issue types produce no finding
    assert classify_discrepancy(_disc(issue_type="po_not_found")) is None


def test_opportunity_tiering():
    def _opp(**kw):
        row = {"opportunity_id": "O-1", "stage": "identified", "financial_impact_gbp": 1200,
               "realised_savings_gbp": None, "supplier_name": "Acme", "deal_id": "D-2",
               "po_id": None, "quote_id": None, "item_description": "notebooks",
               "created_at": datetime.now(timezone.utc)}
        row.update(kw)
        return row
    assert classify_opportunity(_opp())["tier"] == "potential"
    assert classify_opportunity(_opp(stage="negotiation"))["tier"] == "verified"
    assert classify_opportunity(_opp(stage="agreed"))["tier"] == "verified"
    f = classify_opportunity(_opp(stage="realised", realised_savings_gbp=800))
    assert f["tier"] == "verified" and f["recovered_gbp"] == 800.0
    assert classify_opportunity(_opp(stage="rejected")) is None
    assert classify_opportunity(_opp(stage="closed")) is None
    # unknown stage must raise, never be silently dropped (spec rule)
    try:
        classify_opportunity(_opp(stage="mystery"))
        assert False, "unknown stage must raise"
    except ValueError:
        pass
    # finding-dict contract: no "currency" key on any finding leaving this
    # module (build_value_summary strips it from discrepancy findings after
    # FX conversion; opportunity findings must never have carried one)
    assert "currency" not in classify_opportunity(_opp())


def test_dedupe_precedence_and_supersede_flag():
    d = classify_discrepancy(_disc(deal_id="D-9", doc_pk_candidate="INV-9"))
    o = classify_opportunity({"opportunity_id": "O-9", "stage": "agreed",
        "financial_impact_gbp": 950, "realised_savings_gbp": None,
        "supplier_name": "Techworld", "deal_id": "D-9", "po_id": None, "quote_id": None,
        "item_description": None, "created_at": datetime.now(timezone.utc),
        "doc_pk": "INV-9"})
    out = dedupe([o, d])
    kept = [f for f in out if f["superseded_by"] is None]
    supp = [f for f in out if f["superseded_by"] is not None]
    assert len(kept) == 1 and kept[0]["source"] == "discrepancy"
    assert len(supp) == 1 and supp[0]["superseded_by"] == kept[0]["id"]


def test_summarise_totals():
    fs = [
        {"tier": "verified", "amount_gbp": 950.0, "recovered_gbp": 950.0,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 100.0, "recovered_gbp": None,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "potential", "amount_gbp": 500.0, "recovered_gbp": None,
         "supplier_name": "Acme", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 999.0, "recovered_gbp": None,
         "supplier_name": "X", "superseded_by": "other"},   # suppressed: excluded from sums
    ]
    t = summarise(fs)
    assert t["verified_found_gbp"] == 1050.0
    assert t["recovered_gbp"] == 950.0
    assert t["potential_gbp"] == 500.0
    assert t["finding_count"] == 3
    assert t["by_supplier"][0] == {"supplier_name": "Techworld",
                                   "verified_found_gbp": 1050.0, "finding_count": 2}


def test_unconvertible_currency_is_excluded_not_zeroed():
    """A foreign-currency discrepancy with no available FX rate must report an
    honest 'unknown' amount_gbp (None), never a silent £0.00 -- and summarise()
    must not crash and must not count it towards any GBP total."""
    import src.services.value_summary_service as vss

    f = classify_discrepancy(_disc(currency="JPY"))
    assert f["amount_gbp"] == 950.0    # native, pre-conversion

    out = vss._apply_discrepancy_fx(f, rates=None)   # FX unavailable
    assert out["amount_gbp"] is None
    assert out["converted_from"] == {"currency": "JPY", "amount": 950.0, "rate_date": None}
    assert "currency" not in out       # internal staging key must be stripped

    # Mixed with a normal GBP finding: totals must reflect only the known amount.
    gbp_finding = classify_discrepancy(_disc(discrepancy_id=2, doc_pk_candidate="INV-2"))
    t = summarise([out, gbp_finding])
    assert t["verified_found_gbp"] == 950.0    # only the GBP finding counted
    assert t["finding_count"] == 2             # the unconvertible one still counts as found
    supplier_row = t["by_supplier"][0]
    assert supplier_row["supplier_name"] == "Techworld"
    assert supplier_row["finding_count"] == 2
    assert supplier_row["verified_found_gbp"] == 950.0


class _FakeCursor:
    def execute(self, sql, params=()):
        self.description = []

    def fetchall(self):
        return []

    def close(self):
        pass


class FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_source_failure_isolation(monkeypatch):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "_load_discrepancies", lambda cur: (_ for _ in ()).throw(RuntimeError("db")))
    out = vss.build_value_summary(conn=FakeConn())   # FakeConn: cursor() returns a stub whose execute raises for opportunity SQL too if needed
    assert out["sources"]["discrepancies"] == "unavailable"
    assert out["verified_found_gbp"] == 0.0
