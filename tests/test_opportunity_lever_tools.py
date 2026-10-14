"""The saving levers are a set an agent can use: see what exists, read a bucket,
apply one. The agent chooses and explains; the figure comes from the calculation."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orchestration import agentnick_control

_REGISTRY = {
    "price_variance_check": {
        "policy_name": "Price Benchmark Variance", "finding_type": "opportunity",
        "scope": "po_lines", "required_fields": ["supplier_id", "item_id"],
        "default_conditions": {"variance_threshold_pct": 5.0}, "handler": object(),
    },
    "volume_consolidation_check": {
        "policy_name": "Volume Consolidation", "finding_type": "opportunity",
        "scope": "po_lines", "required_fields": ["minimum_volume_gbp"],
        "default_conditions": {}, "handler": object(),
    },
}


class _Miner:
    def __init__(self, findings=(), registry=_REGISTRY):
        self.findings, self.registry, self.ran = list(findings), registry, []

    def _get_policy_registry(self):
        if isinstance(self.registry, Exception):
            raise self.registry
        return self.registry

    def execute(self, ctx):
        self.ran.append(ctx)
        return SimpleNamespace(status=SimpleNamespace(value="success"), error=None,
                               data={"findings": self.findings,
                                     "policy_events": [{"message": "threshold not met"}]})


def _tools(miner):
    nick = SimpleNamespace(agents={"opportunity_miner": miner} if miner else {},
                           auto_registry=None)
    return {t.name: t for t in agentnick_control.build_tools(nick, workflow_id="wf-1",
                                                             user_id="u-1")}


def test_the_levers_are_on_offer_as_tools():
    assert {"list_opportunity_levers", "get_opportunity_bucket",
            "evaluate_opportunity_lever"} <= set(_tools(_Miner()))


def test_without_the_miner_there_are_no_lever_tools():
    assert "list_opportunity_levers" not in _tools(None)


def test_the_catalogue_is_the_rule_book_not_a_list_in_the_tool():
    levers = _tools(_Miner())["list_opportunity_levers"].handler()["levers"]

    assert [l["lever"] for l in levers] == ["price_variance_check",
                                            "volume_consolidation_check"]
    assert levers[0]["name"] == "Price Benchmark Variance"
    assert levers[0]["needs"] == ["supplier_id", "item_id"]
    assert levers[0]["thresholds"] == {"variance_threshold_pct": 5.0}


def test_an_unreadable_catalogue_is_not_an_empty_one():
    out = _tools(_Miner(registry=RuntimeError("rule book down")))[
        "list_opportunity_levers"].handler()

    assert out["levers"] is None and "rule book down" in out["note"]


def test_applying_a_lever_runs_only_that_lever_on_that_supplier_and_item():
    miner = _Miner(findings=[{"financial_impact_gbp": 300.0, "supplier_id": "SUP-1",
                              "item_id": "ITM-1", "calculation_details": {"quantity": 40},
                              "source_records": ["PO-1"]}])

    out = _tools(miner)["evaluate_opportunity_lever"].handler(
        lever="price_variance_check", supplier_id="SUP-1", item_id="ITM-1")

    (ctx,) = miner.ran
    assert ctx.input_data["workflow"] == "price_variance_check"
    assert ctx.input_data["conditions"] == {"supplier_id": "SUP-1", "item_id": "ITM-1"}
    assert (ctx.workflow_id, ctx.user_id) == ("wf-1", "u-1")
    assert out["found_a_saving"] is True
    assert out["results"][0]["saving_gbp"] == 300.0
    assert out["results"][0]["working"] == {"quantity": 40}


def test_a_lever_that_finds_nothing_says_why():
    out = _tools(_Miner())["evaluate_opportunity_lever"].handler(
        lever="price_variance_check", supplier_id="SUP-1")

    assert out["found_a_saving"] is False and out["results"] == []
    assert out["events"] == [{"message": "threshold not met"}]


def test_an_invented_lever_is_not_run_and_the_real_ones_are_offered():
    miner = _Miner()

    out = _tools(miner)["evaluate_opportunity_lever"].handler(
        lever="negotiate_harder", supplier_id="SUP-1")

    assert miner.ran == []
    assert out["found"] is False
    assert out["available"] == ["price_variance_check", "volume_consolidation_check"]


def test_no_bucket_is_not_reported_as_no_saving(monkeypatch):
    from src.services import opportunity_buckets

    monkeypatch.setattr(opportunity_buckets, "get_bucket", lambda s, i: None)

    out = _tools(_Miner())["get_opportunity_bucket"].handler(supplier_id="S", item_id="I")

    assert out["bucket"] is None and "not the same" in out["note"]
