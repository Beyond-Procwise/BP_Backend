"""How a proposal's price changes over its term (price_schedule.py).

The text is Meridia's MCP-Q-7740 (V3) order form as the parser stored it. Extraction kept only its
Year 1 figures; the schedule rises ~5% a year under a stated 3.5% uplift, which the supplier's own
note puts at "roughly £53,000" over the term.
"""
import json

from src.services.extraction.price_schedule import (
    attach_schedules, escalation_findings, finding_notes, pricing_terms, schedules,
)

MCP_V3 = """## Sheet: Order Form

| M | Meridia Cloud Platforms Ltd |  | Order Form |  |
| --- | --- | --- | --- | --- |
| Milton Keynes |  | Term:  36 months (3 years) |  |  |
| Buckinghamshire |  | Annual uplift (clause 6.2):  3.5% per annum |  |  |
| 3-year subscription schedule  (uplift applied to Years 2–3) |  |  |  |  |
| Line item | Year 1 (£) | Year 2 (£) | Year 3 (£) | 3-yr subtotal (£) |
| Platform licence — Enterprise (240 seats) | 1008000 | 1058400 | 1111320 | 3177720 |
| Premium support & SLA uplift | 94000 | 98700 | 103635 | 296335 |
| Sandbox / non-prod environment | 42000 | 44100 | 46305 | 132405 |
| Implementation & onboarding (one-off, Year 1) | 56000 | 0 | 0 | 56000 |
| Total contract value (TCV), 3 years |  |  |  | 3662460 |
|  |  | Year 1 total (ex-VAT) |  | 1200000 |
| RISK — Clause 6.2 states a 3.5% annual uplift, but the Year 2 and Year 3 figures compound at ~5.0%. |  |  |  |  |
| Commercial terms |  |  |  |  |
| Term | 36 months, annual in advance |  |  |  |
| Uplift | Clause 6.2 — 3.5% per annum (see schedule) |  |  |  |
"""


def test_reads_each_line_by_year_and_the_stated_contract_value():
    s = schedules(MCP_V3)
    assert [l["description"] for l in s["lines"]] == [
        "Platform licence — Enterprise (240 seats)", "Premium support & SLA uplift",
        "Sandbox / non-prod environment", "Implementation & onboarding (one-off, Year 1)"]
    assert [p["amount"] for p in s["lines"][0]["periods"]] == [1008000, 1058400, 1111320]
    assert [p["label"] for p in s["lines"][0]["periods"]] == ["Year 1", "Year 2", "Year 3"]
    assert s["lines"][0]["term_total"] == 3177720
    assert s["stated_tcv"] == 3662460
    assert sum(l["term_total"] for l in s["lines"]) == s["stated_tcv"]


def test_reads_the_term_and_the_uplift_the_document_states():
    t = pricing_terms(MCP_V3)
    assert t["term_months"] == 36
    assert t["uplift_pct"] == 3.5
    assert t["uplift_text"] == "Annual uplift (clause 6.2):  3.5% per annum"
    assert t["uplift_conflict"] is False


def test_an_index_plus_margin_is_indexation_and_commentary_is_not_a_term():
    t = pricing_terms("| Uplift | CPI + 4.5% per annum |\n"
                      "| RISK — the uplift has worsened from CPI+2.0% to CPI+4.5%. |\n"
                      "| Buckinghamshire | Annual uplift (clause 6.2):  6.5% per annum |")
    assert t["indexation"] == "CPI + 4.5%"
    assert t["uplift_pct"] == 6.5
    assert t["uplift_conflict"] is False


def test_two_different_fixed_rates_are_left_for_a_person():
    t = pricing_terms("| Annual uplift: 3.5% per annum |\n| Uplift | 5% per annum |")
    assert t["uplift_pct"] is None and t["uplift_conflict"] is True


def test_a_schedule_rising_faster_than_the_stated_uplift_is_a_finding_with_its_cost():
    s, t = schedules(MCP_V3), pricing_terms(MCP_V3)
    f = escalation_findings(s["lines"], t, tolerance_pp=0.25)
    assert [x["issue_type"] for x in f] == ["uplift_above_stated"]
    assert [round(x["actual_pct"], 1) for x in f[0]["lines"]] == [5.0, 5.0, 5.0]
    # The supplier's own note: "roughly £53,000". Against Year 1 compounded at 3.5%.
    assert round(f[0]["extra"]) == 52939
    assert "rises faster than the stated 3.5% uplift" in finding_notes(f[0])
    assert "£52,939" in finding_notes(f[0])


def test_a_one_off_charge_is_not_a_rise_and_a_schedule_within_its_uplift_is_quiet():
    lines = [{"description": "Licence", "periods": [{"period": 1, "amount": 100.0}, {"period": 2, "amount": 103.5}]},
             {"description": "Onboarding", "periods": [{"period": 1, "amount": 50.0}, {"period": 2, "amount": 0.0}]}]
    assert escalation_findings(lines, {"uplift_pct": 3.5}, 0.25) == []


def test_rises_with_no_uplift_stated_are_a_finding_unless_an_index_is_named():
    lines = [{"description": "Licence", "periods": [{"period": 1, "amount": 100.0}, {"period": 2, "amount": 110.0}]}]
    f = escalation_findings(lines, {"uplift_pct": None, "indexation": None}, 0.25)
    assert f[0]["issue_type"] == "price_rises_unstated" and f[0]["extra"] == 10
    assert escalation_findings(lines, {"uplift_pct": None, "indexation": "CPI"}, 0.25) == []


def test_line_items_get_their_schedule_by_description():
    s = schedules(MCP_V3)
    items = attach_schedules([{"item_description": "Platform licence — Enterprise (240 seats)", "line_total": 1008000},
                              {"item_description": "Something else"}], s["lines"])
    stored = json.loads(items[0]["price_schedule"])          # JSON text, as the line tables hold it
    assert stored["term_total"] == 3177720
    assert [p["amount"] for p in stored["periods"]] == [1008000, 1058400, 1111320]
    assert "price_schedule" not in items[1]


def test_a_document_with_no_schedule_has_nothing_to_say():
    fsm = "| Description | UoM | Qty | Rate (£) | Amount (£) |\n| Service Desk (24x7, 2,400 users) — annual | year | 1 | 540000 | 540000 |"
    assert schedules(fsm)["lines"] == []
    assert pricing_terms(fsm)["uplift_pct"] is None
