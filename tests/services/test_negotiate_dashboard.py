from __future__ import annotations

import datetime as dt

from src.services import negotiate_dashboard as nd


class _FakeCur:
    """Returns canned rows for the first script entry whose substring is in the
    SQL. raise_on simulates a missing table (e.g. supplier_response)."""
    def __init__(self, script, raise_on=()):
        self.script = script
        self.raise_on = raise_on
        self._rows = []
        self.description = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        for sub in self.raise_on:
            if sub.lower() in s.lower():
                raise RuntimeError("relation does not exist")
        for sub, rows in self.script:
            if sub.lower() in s.lower():
                self._rows = [tuple(r.values()) for r in rows]
                self.description = [(k,) for k in (rows[0].keys() if rows else [])]
                return
        self._rows = []
        self.description = None

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_money_formatting_matches_design_style():
    assert nd._money(6750, "GBP") == "£6.8K"
    assert nd._money(544000, "GBP") == "£544K"
    assert nd._money(1_200_000, "GBP") == "£1.2M"
    assert nd._money(500, "GBP") == "£500"
    assert nd._money(0, "GBP") == "£0"
    assert nd._money(None) == ""
    assert nd._money(1000, "USD") == "$1.0K"


def _deal(**over):
    base = {"deal_id": "D1", "deal_name": "Acme Deal", "supplier_id": "SUP-A",
            "supplier_name": "Acme", "currency": "GBP", "quote_total": 1000.0,
            "po_total": 900.0, "invoice_total": None, "price_variance_pct": 10.0,
            "last_activity_date": dt.date(2026, 6, 11), "deal_date": None,
            "first_activity_date": None, "quote_count": 1, "po_count": 1,
            "invoice_count": 0, "cycle_days_quote_to_po": None,
            "cycle_days_po_to_invoice": None}
    base.update(over)
    return base


def test_proposal_snapshot_computes_savings_and_shape():
    cur = _FakeCur(script=[
        ("from proc.bp_deal_overview", [_deal()]),
        ("select payment_terms from proc.bp_purchase_order_trgt", [{"payment_terms": "30 Days"}]),
        ("from proc.bp_contracts", []),
    ])
    snap = nd.proposal_snapshot(cur, "D1")
    by = {x["label"]: x["value"] for x in snap}
    assert by["tvc"] == "£900"          # actual PO total is the contract value
    assert by["savingsVsBaseline"] == "10%"   # (1000-900)/1000
    assert by["paymentTerms"] == "30 Days"
    assert [x["label"] for x in snap] == ["tvc", "annualRunRate", "savingsVsBaseline",
                                          "paymentTerms", "Terms"]


def test_cost_over_time_is_cumulative_and_sorted():
    cur = _FakeCur(script=[
        ("quote_date dt", [{"dt": dt.date(2024, 1, 1), "amt": 1000}]),
        ("order_date dt", [{"dt": dt.date(2024, 2, 1), "amt": 900}]),
        ("invoice_date dt", [{"dt": dt.date(2024, 3, 1), "amt": 950}]),
    ])
    series = nd.cost_over_time(cur, "D1")
    assert [p["time"] for p in series] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert series[0]["Overall"] == 1000.0
    assert series[-1]["Overall"] == 2850.0   # 1000 + 900 + 950 cumulative
    assert series[-1]["PO"] == 900.0 and series[-1]["Invoice"] == 950.0


def test_proposal_summary_baseline_vs_current_diff():
    cur = _FakeCur(script=[
        ("from proc.bp_deal_overview", [_deal()]),
        ("from proc.bp_quote_line_items_trgt", [{"item": "Router A", "u": 55.0}]),
        ("from proc.bp_po_line_items_trgt", [{"item": "Router A", "u": 51.8}]),
    ])
    rows = nd.proposal_summary(cur, "D1")
    r = next(x for x in rows if x["item"] == "Router A")
    assert r["baselineUnit"] == "£55"
    assert r["currentUnit"] == "£52"
    assert r["unit"].startswith("-")   # price dropped


def test_offer_history_empty_when_supplier_response_absent():
    cur = _FakeCur(script=[("from proc.bp_deal_overview", [_deal()])],
                   raise_on=("supplier_response",))
    assert nd.offer_version_history(cur, "D1") == []


def _insights_for(risk_score):
    cur = _FakeCur(script=[("from proc.bp_supplier",
                            [{"is_preferred_supplier": False,
                              "risk_score": risk_score}])])
    return nd._supplier_insights(cur, {"supplier_id": "SUP-1"})


def test_supplier_risk_is_judged_on_the_0_to_100_scale():
    # risk_score is VARCHAR on a 0-100 scale (min 5, median 49.57, max 94.94).
    # A median supplier is NOT elevated risk; testing against 0.6 made all 5000
    # of them elevated, so every deal got the same "de-risk" recommendation.
    _, key_driver, recommendation = _insights_for("49.57")
    assert key_driver == "Revenue growth"
    assert "De-risk" not in recommendation


def test_supplier_risk_above_the_bar_still_reads_as_elevated():
    _, key_driver, recommendation = _insights_for("94.94")
    assert key_driver == "Risk and stability"
    assert "De-risk" in recommendation


def test_build_returns_none_for_unknown_deal():
    cur = _FakeCur(script=[("from proc.bp_deal_overview", [])])

    class _Conn:
        def cursor(self_inner):
            return cur

    assert nd.build_negotiate_dashboard("NOPE", conn=_Conn()) is None


def test_build_full_shape_for_known_deal():
    cur = _FakeCur(script=[("from proc.bp_deal_overview", [_deal()])],
                   raise_on=("supplier_response",))

    class _Conn:
        def cursor(self_inner):
            return cur

    out = nd.build_negotiate_dashboard("D1", conn=_Conn())
    assert set(out) >= {"summary", "proposalSnapshot", "versionHistory", "negotiationData",
                        "negotiationStrategy", "costOverTime", "volumeTrend",
                        "proposalSummary", "demandVsVolumeData"}
    assert out["versionHistory"] == []          # safe fallback
    assert isinstance(out["negotiationStrategy"], list) and out["negotiationStrategy"]
    assert out["summary"].startswith("Deal 'Acme Deal'")
