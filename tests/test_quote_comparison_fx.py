"""Quote comparison never invents an exchange rate.

A quote whose currency has no rate cannot be put on the same footing as the
others, so it is marked "not assessed" and left out of the ranking — it is not
assumed 1:1 against sterling, and no hardcoded USD rate is used.
"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentStatus
from agents.quote_comparison_agent import QuoteComparisonAgent
from tests.test_quote_comparison_agent import DummyNick, _build_context


def _quote(name, sid, cost, currency):
    return {
        "name": name, "supplier_id": sid, "total_spend": cost, "total_cost": cost,
        "volume": 1, "tenure": "Net 30", "quote_file_s3_path": None, "currency": currency,
    }


def _agent(monkeypatch, batch):
    """An agent whose live rate batch is ``batch`` (USD-quoted, units per 1 USD)."""
    monkeypatch.setattr("repositories.fx_rate_repo.get_or_refresh_rates", lambda *a, **k: batch)
    agent = QuoteComparisonAgent(DummyNick())
    monkeypatch.setattr(agent, "_read_table", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("rates come from fx_rate_repo, not ad-hoc tables")))
    return agent


def _run(agent, quotes):
    ctx = _build_context(quotes, {"supplier_ids": [q["supplier_id"] for q in quotes]})
    result = agent.run(ctx)
    assert result.status == AgentStatus.SUCCESS
    rows = {r["supplier_id"]: r for r in result.data["comparison"] if r["name"] != "weighting"}
    return rows, result.data.get("recommended_quote")


def test_unknown_currency_is_not_assessed_and_never_recommended(monkeypatch):
    agent = _agent(monkeypatch, {"rates": {"GBP": 0.75, "USD": 1.0}})
    # The NZD quote is numerically the cheapest: assumed 1:1 it would win.
    rows, rec = _run(agent, [_quote("A", "S1", 100, "GBP"), _quote("B", "S2", 10, "NZD")])
    assert rows["S2"]["total_cost_gbp"] is None
    assert rows["S2"]["fx_status"] == "not_assessed"
    assert "NZD" in rows["S2"]["fx_note"]
    assert rows["S2"]["recommendation"]["ticker"] == "NOT_ASSESSED"
    assert rec["supplier_id"] == "S1"


def test_known_currency_converts_from_the_live_batch(monkeypatch):
    agent = _agent(monkeypatch, {"rates": {"GBP": 0.75, "USD": 1.0}})
    rows, _ = _run(agent, [_quote("A", "S1", 100, "GBP"), _quote("B", "S2", 100, "USD")])
    assert rows["S2"]["total_cost_gbp"] == pytest.approx(75.0)
    assert rows["S2"].get("fx_status") != "not_assessed"


def test_no_hardcoded_usd_rate_when_rates_are_unavailable(monkeypatch):
    agent = _agent(monkeypatch, None)
    rows, rec = _run(agent, [_quote("A", "S1", 100, "GBP"), _quote("B", "S2", 100, "USD")])
    assert rows["S2"]["total_cost_gbp"] is None   # not 79.0
    assert rows["S2"]["fx_status"] == "not_assessed"
    assert rec["supplier_id"] == "S1"


def test_gbp_needs_no_rate_lookup(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("GBP is identity; no lookup needed")
    monkeypatch.setattr("repositories.fx_rate_repo.get_or_refresh_rates", boom)
    agent = QuoteComparisonAgent(DummyNick())
    rows, rec = _run(agent, [_quote("A", "S1", 100, "GBP"), _quote("B", "S2", 120, "GBP")])
    assert rows["S1"]["total_cost_gbp"] == 100
    assert rec["supplier_id"] == "S1"


def test_nothing_comparable_means_no_recommendation(monkeypatch):
    agent = _agent(monkeypatch, {"rates": {"GBP": 0.75, "USD": 1.0}})
    rows, rec = _run(agent, [_quote("A", "S1", 10, "NZD"), _quote("B", "S2", 20, "CHF")])
    assert rec is None
    assert all(r["fx_status"] == "not_assessed" for r in rows.values())
