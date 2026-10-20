"""Compliance "Invoice > PO variance" is the triage engine's own answer.

The figure is read from the open cumulative_total findings, so the Compliance
screen and the Action Centre cannot disagree about the same invoice, and the
tolerance that decided them is the governed one, not a number baked into SQL.
"""
import os

import pytest

from src.services.compliance.po_variance import summarise


def f(delta, deal="D1"):
    return {"finding_id": 1, "deal_id": deal, "delta": delta}


def test_counts_cases_and_sums_the_gbp_figures():
    out = summarise([f("£950.00"), f("£1,687.57 (2,230.94 USD)")], invoiced_pos=10)
    assert out["cases"] == 2
    assert out["value_at_risk_gbp"] == pytest.approx(2637.57)
    assert out["excluded_no_gbp"] == 0
    assert out["pct_of_invoiced_pos"] == 20.0


def test_a_finding_with_no_gbp_figure_is_counted_but_not_summed():
    # triage writes the native amount with no leading £ when it had no FX rate
    out = summarise([f("£100.00"), f("315.21 USD (no FX rate)")], invoiced_pos=4)
    assert out["cases"] == 2
    assert out["value_at_risk_gbp"] == pytest.approx(100.0)
    assert out["excluded_no_gbp"] == 1


def test_no_gbp_figure_at_all_is_none_not_zero():
    out = summarise([f("315.21 USD (no FX rate)")], invoiced_pos=4)
    assert out["cases"] == 1
    assert out["value_at_risk_gbp"] is None


def test_no_findings_is_a_measured_zero():
    out = summarise([], invoiced_pos=50)
    assert out["cases"] == 0
    assert out["value_at_risk_gbp"] == 0
    assert out["pct_of_invoiced_pos"] == 0.0


def test_no_invoiced_pos_means_no_percentage():
    out = summarise([], invoiced_pos=0)
    assert out["pct_of_invoiced_pos"] is None


def test_usd_figure_converts_gbp_with_the_usd_quoted_batch():
    # rates are units per 1 USD: GBP 0.75 means 1 GBP = 1/0.75 USD
    out = summarise([f("£75.00")], invoiced_pos=1, rates={"GBP": 0.75, "USD": 1.0})
    assert out["value_at_risk_usd"] == pytest.approx(100.0)


def test_usd_figure_is_none_without_a_gbp_rate_not_a_guess():
    out = summarise([f("£75.00")], invoiced_pos=1, rates=None)
    assert out["value_at_risk_usd"] is None
    assert out["value_at_risk_gbp"] == pytest.approx(75.0)


def test_usd_figure_of_no_cases_is_zero():
    assert summarise([], invoiced_pos=1, rates=None)["value_at_risk_usd"] == 0


def test_tolerance_does_not_name_an_internal_table(monkeypatch):
    # the output-safety filter withholds any field that names a table or route
    from src.services.compliance import po_variance as mod
    out = mod._tolerance()
    assert out is None or "source" not in out


@pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                    reason="needs the live database (PROCWISE_TEST_LIVE_DB=1)")
def test_live_cases_equal_the_action_centres_open_overbilling_findings():
    from src.services.compliance.po_variance import po_variance
    from src.services.db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_detection_finding "
                    "WHERE rule_id='cumulative_total' AND status='open'")
        open_findings = cur.fetchone()[0]
    assert po_variance()["cases"] == open_findings
