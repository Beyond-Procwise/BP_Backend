from __future__ import annotations

import datetime as dt

import pytest

from src.services import opportunity_dashboard as od
from src.services import opportunity_store as ostore


class _FakeCur:
    def __init__(self, script):
        self.script = script
        self._rows = []
        self.description = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
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


def test_money_and_pct_change():
    assert od._money(181000) == "£181k"
    assert od._money(1_200_000) == "£1.2M"
    assert od._money(5400) == "£5.4k"
    assert od._money(0) == "£0"
    assert od._pct_change(110, 100) == "+10%"
    assert od._pct_change(90, 100) == "-10%"
    assert od._pct_change(5, 0) == "+100%"
    assert od._pct_change(0, 0) == "+0%"


def test_opportunities_data_kpis():
    cur = _FakeCur(script=[
        ("count(*) total", [{"total": 28, "identified": 18, "closed": 10, "in_flight": 9,
                             "potential": 1_200_000, "realised": 420000, "cat_impact": 4}]),
        ("cur_n", [{"cur_n": 11, "prev_n": 10, "cur_p": 110, "prev_p": 100,
                    "cur_r": 8, "prev_r": 0, "cur_if": 4, "prev_if": 0}]),
    ])
    d = od.opportunities_data(cur)
    assert d["totalOpportunity"] == 28
    assert d["identified"] == 18 and d["closed"] == 10 and d["inFlight"] == 9
    assert d["potential"] == "£1.2M"
    assert d["realised"] == "£420k"
    assert d["categoryImpact"] == 4
    assert d["opportunitiesChange"] == "+10%"
    assert d["potentialChange"] == "+10%"


def test_savings_pipeline_shape():
    cur = _FakeCur(script=[
        ("coalesce(sum(financial_impact_gbp),0) identified",
         [{"identified": 1_200_000, "negotiation": 800000, "agreed": 520000, "realised": 420000}]),
    ])
    p = od.savings_pipeline(cur)
    assert [x["label"] for x in p] == ["opportunities Identified", "negotiation Started",
                                       "savings Agreed", "savings Realized"]
    assert p[0]["value"] == "£1.2M" and p[-1]["value"] == "£420k"


def test_detailed_opportunities_mapping_with_uncategorised_fallback():
    cur = _FakeCur(script=[
        ("from proc.bp_opportunity", [{
            "opportunity_id": "9", "detected_on": dt.datetime(2026, 6, 8, 12, 0),
            "detector_type": "Maverick Spend Detection", "category_id": None,
            "supplier_name": "City of Newport", "supplier_id": "SUP-X",
            "item_description": "Tier 3 Marketing Services", "item_id": None,
            "financial_impact_gbp": 100000, "stage": "identified"}]),
    ])
    rows = od.detailed_opportunities(cur)
    r = rows[0]
    assert r == {
        "opportunityId": "9", "date": "2026-06-08", "type": "Maverick Spend Detection",
        "category": "Uncategorised", "supplier": "City of Newport",
        "opportunity": "Tier 3 Marketing Services", "potentialSaving": "£100k",
        "stage": "identified",
    }


def test_savings_identified_vs_completed_sorted():
    cur = _FakeCur(script=[
        ("from proc.bp_opportunity where detected_on is not null group by 1,2",
         [{"mon": "Feb", "m": dt.datetime(2026, 2, 1), "v": 640000},
          {"mon": "Jan", "m": dt.datetime(2026, 1, 1), "v": 210000}]),
        ("stage='realised' and stage_updated_at is not null",
         [{"mon": "Jan", "v": 120000}]),
    ])
    out = od.savings_identified_vs_completed(cur)
    assert [x["month"] for x in out] == ["Jan", "Feb"]   # chronological
    assert out[0] == {"month": "Jan", "identified": 210000, "completed": 120000}
    assert out[1]["completed"] == 0   # Feb has no realised


def test_build_returns_all_keys():
    cur = _FakeCur(script=[])

    class _Conn:
        def cursor(self_inner):
            return cur

    out = od.build_opportunities_dashboard(conn=_Conn())
    assert set(out) == {"opportunitiesData", "savingsPipeline",
                        "savingsIdentifiedVsCompleted", "opportunityTrends",
                        "detailedOpportunities"}


def test_set_stage_rejects_invalid_stage():
    with pytest.raises(ValueError):
        ostore.set_stage("9", "bogus", conn=object())
