"""The daily brief's readings. Pure functions on rows — no database.

Row shapes mirror proc.bp_extraction_discrepancy (joined to its deal),
proc.bp_decision, proc.bp_opportunity, proc.workflow_email_tracking and the
contract expiry rule's alerts.
"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from src.services.daily_brief import sections

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def ago(days: int) -> datetime:
    return NOW - timedelta(days=days)


def issue(i, days, severity="critical", deal="D1", **more):
    return {"discrepancy_id": i, "doc_pk_candidate": f"INV-{i}", "issue_type": "duplicate_invoice",
            "severity": severity, "created_at": ago(days), "deal_id": deal,
            "deal_name": f"Deal {deal}", "supplier_name": "Acme", **more}


# ── momentum ─────────────────────────────────────────────────────────────────

def test_momentum_reports_this_window_against_the_one_before():
    got = sections.momentum(
        window_days=7,
        issues={"raised_current": 79, "raised_prior": 4, "closed_current": 276, "closed_prior": 0},
        decisions={"raised_current": 10, "raised_prior": 3, "made_current": 9, "made_prior": 2},
        recovered={"current": 226.78, "prior": 0}, opportunities={"current": 0, "prior": 0})
    by_key = {m["key"]: m for m in got["metrics"]}
    assert got["windowDays"] == 7
    assert (by_key["issues_closed"]["current"], by_key["issues_closed"]["prior"]) == (276, 0)
    assert by_key["recovered"] == {"key": "recovered", "unit": "gbp", "current": 226.78, "prior": 0.0}
    # a counted zero is news, and is kept
    assert by_key["opportunities_progressed"]["current"] == 0


def test_momentum_drops_a_measure_whose_source_did_not_answer():
    got = sections.momentum(window_days=7, issues=None, decisions=None,
                            recovered={"current": 5, "prior": 1}, opportunities=None)
    assert [m["key"] for m in got["metrics"]] == ["recovered"]


def test_momentum_is_absent_when_nothing_answered():
    assert sections.momentum(window_days=7, issues=None, decisions=None,
                             recovered=None, opportunities=None) is None


# ── critical and aged ────────────────────────────────────────────────────────

def aged(issues, priced=None, **kw):
    args = {"now": NOW, "min_days": 14, "severities": ["critical"], "top_n": 5, **kw}
    return sections.aged_issues(issues, priced or {}, **args)


def test_an_issue_younger_than_the_cut_off_is_not_aged():
    got = aged([issue(1, 13), issue(2, 14), issue(3, 40)])
    assert got["count"] == 2
    assert got["oldestDays"] == 40
    assert [i["name"] for i in got["items"]] == ["INV-3", "INV-2"]


def test_the_cut_off_is_the_setting_not_a_constant():
    rows = [issue(1, 13), issue(2, 14), issue(3, 40)]
    assert aged(rows, min_days=30)["count"] == 1
    assert aged(rows, min_days=0)["count"] == 3


def test_only_the_chosen_grades_count():
    rows = [issue(1, 30), issue(2, 30, severity="warning")]
    assert aged(rows)["count"] == 1
    assert aged(rows, severities=["critical", "warning"])["count"] == 2


def test_aged_issues_are_named_largest_first_then_oldest():
    rows = [issue(1, 90), issue(2, 20), issue(3, 20), issue(4, 60)]
    priced = {2: {"amount_gbp": 500.0}, 3: {"amount_gbp": 9000.0}}
    got = aged(rows, priced)
    assert [i["name"] for i in got["items"]] == ["INV-3", "INV-2", "INV-1", "INV-4"]
    assert got["valueGbp"] == 9500.0
    assert got["valuePartial"] is True       # two of the four have no readable amount


def test_money_counted_under_another_issue_is_not_named_or_added_again():
    """A line under an overbilled order is the order's money seen again. It is
    still an open issue, so it is counted; it is not listed and not summed."""
    rows = [issue(1, 30), issue(2, 30)]
    priced = {1: {"amount_gbp": 20000.0}, 2: {"amount_gbp": 20000.0, "superseded_by": "disc:1"}}
    got = aged(rows, priced)
    assert got["count"] == 2
    assert got["valueGbp"] == 20000.0
    assert [i["name"] for i in got["items"]] == ["INV-1"]
    assert got["valuePartial"] is False


def test_the_list_is_capped_and_the_count_is_not():
    got = aged([issue(i, 30) for i in range(12)], top_n=3)
    assert len(got["items"]) == 3 and got["count"] == 12


def test_no_aged_issues_is_absent_not_zero():
    assert aged([issue(1, 2)]) is None


def test_aged_decisions_counts_by_kind_past_the_cut_off():
    rows = [{"subject_type": "authorization", "created_at": ago(30)},
            {"subject_type": "authorization", "created_at": ago(8)},
            {"subject_type": "policy_conflict", "created_at": ago(9)},
            {"subject_type": "email_reply", "created_at": ago(6)}]
    got = sections.aged_decisions(rows, now=NOW, min_days=7)
    assert got["count"] == 3 and got["oldestDays"] == 30
    assert got["byType"] == [{"type": "authorization", "count": 2}, {"type": "policy_conflict", "count": 1}]
    assert sections.aged_decisions(rows, now=NOW, min_days=31) is None


def test_critical_aged_carries_only_the_parts_that_exist():
    assert sections.critical_aged(issues=None, decisions=None,
                                  expired_contracts={"count": 0}, min_days=14) is None
    got = sections.critical_aged(issues=None, decisions={"count": 2},
                                 expired_contracts={"count": 3}, min_days=14)
    assert set(got) == {"minDays", "decisions", "expiredContracts"}


# ── upcoming ─────────────────────────────────────────────────────────────────

def alert(cid, days, suppressed=None):
    return SimpleNamespace(contract_id=cid, days_to_end=days, suppressed_by=suppressed,
                           end_date=date(2026, 10, 13) + timedelta(days=days), bucket="0-1")


CONTRACTS = {"A": {"name": "A", "gbp": 100.0}, "B": {"name": "B", "gbp": 5000.0},
             "C": {"name": "C", "gbp": None}, "D": {"name": "D", "gbp": 900000.0},
             "E": {"name": "E", "gbp": 1.0}}


def up(alerts, **kw):
    args = {"horizon_days": 90, "top_n": 5, "min_value_gbp": None, **kw}
    return sections.upcoming(alerts=alerts, contracts=CONTRACTS, name_of=lambda r: r["name"],
                             gbp_of=lambda r: r["gbp"], **args)


def test_upcoming_names_the_largest_contracts_inside_the_horizon():
    got = up([alert("A", 10), alert("B", 80), alert("C", 5), alert("D", 91), alert("E", -3)])
    assert got["count"] == 3                      # D is past the horizon, E already ended
    assert [i["name"] for i in got["items"]] == ["B", "A", "C"]
    assert got["valueGbp"] == 5100.0 and got["valuePartial"] is True
    assert got["nearestDays"] == 5
    assert got["items"][0]["due"] == "2027-01-01"


def test_the_horizon_is_the_setting():
    assert up([alert("A", 10), alert("D", 91)], horizon_days=120)["count"] == 2


def test_a_contract_a_demand_already_covers_is_not_upcoming_news():
    assert up([alert("A", 10, suppressed="DEM-1")]) is None


def test_a_value_floor_keeps_only_what_clears_it_and_says_what_it_could_not_judge():
    got = up([alert("A", 10), alert("B", 20), alert("C", 5)], min_value_gbp=1000)
    assert [i["name"] for i in got["items"]] == ["B"]
    assert got["count"] == 1 and got["unvalued"] == 1 and got["minValueGbp"] == 1000


# ── approvals ────────────────────────────────────────────────────────────────

def test_approvals_counts_every_open_decision_by_kind():
    rows = [{"subject_type": "authorization", "created_at": ago(27)},
            {"subject_type": "email_reply", "created_at": ago(74)}]
    got = sections.approvals(rows, now=NOW)
    assert got["count"] == 2 and got["oldestDays"] == 74
    assert sections.approvals([], now=NOW) is None


# ── pipeline ─────────────────────────────────────────────────────────────────

def pipe(issues, **kw):
    args = {"now": NOW, "severities": ["critical", "warning"], "top_n": 8, "deals_in_scope": 16, **kw}
    return sections.pipeline(issues, **args)


def test_pipeline_groups_open_issues_by_deal_worst_first():
    rows = [issue(1, 3, deal="D1", severity="warning"), issue(2, 9, deal="D1", severity="warning"),
            issue(3, 1, deal="D2"), issue(4, 1, deal=None)]
    got = pipe(rows)
    assert got["deals"] == 16 and got["dealsNeedingAttention"] == 2 and got["openIssues"] == 4
    assert [d["dealId"] for d in got["items"]] == ["D2", "D1"]      # a critical outranks two warnings
    assert got["items"][1] == {"dealId": "D1", "name": "Deal D1", "supplier": "Acme",
                               "issues": 2, "critical": 0, "oldestDays": 9}


def test_pipeline_ignores_grades_the_settings_leave_out():
    assert pipe([issue(1, 3, severity="info")])["openIssues"] == 0


def test_a_clean_pipeline_is_reported_and_no_pipeline_is_absent():
    assert pipe([])["dealsNeedingAttention"] == 0
    assert pipe([], deals_in_scope=0) is None


# ── supplier emails ──────────────────────────────────────────────────────────

def test_supplier_emails_lists_replies_newest_first():
    rows = [{"supplier_id": "SUP-A", "supplier_name": "A Ltd", "subject": "Re: one", "responded_at": ago(5)},
            {"supplier_id": "SUP-B", "supplier_name": None, "subject": "Re: two", "responded_at": ago(1)},
            {"supplier_id": "SUP-C", "subject": "never answered", "responded_at": None}]
    got = sections.supplier_emails(rows, 1, now=NOW, window_days=14)
    assert got["replies"] == 2 and got["awaitingDecision"] == 1
    assert got["items"] == [{"supplier": "SUP-B", "subject": "Re: two", "daysAgo": 1},
                            {"supplier": "A Ltd", "subject": "Re: one", "daysAgo": 5}]
    assert sections.supplier_emails([], 0, now=NOW, window_days=14) is None


# ── opportunities ────────────────────────────────────────────────────────────

def test_opportunities_are_named_largest_first_and_summed_honestly():
    rows = [{"opportunity_id": 1, "item_description": "small", "financial_impact_gbp": 10},
            {"opportunity_id": 2, "item_description": "big", "financial_impact_gbp": 5000},
            {"opportunity_id": 3, "detector_type": "price_variance", "financial_impact_gbp": None}]
    got = sections.opportunities(rows, top_n=2)
    assert got["count"] == 3 and got["valueGbp"] == 5010.0 and got["valuePartial"] is True
    assert [i["name"] for i in got["items"]] == ["big", "small"]
    assert sections.opportunities([], top_n=5) is None
