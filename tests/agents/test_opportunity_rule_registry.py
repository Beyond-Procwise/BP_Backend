"""The opportunity detector registry is driven by proc.bp_rule, not bp_policy.

Detection rules and authorization policies are separate concerns kept in
separate tables. These tests hold that line: what runs, and with what
thresholds, comes from the rule book -- and a policy row cannot reach in and
change it no matter what it is called.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.opportunity_miner_agent import OpportunityMinerAgent
from engines.policy_engine import PolicyEngine
from engines.rule_book import RuleBook, RuleBookUnavailable


def _rule(rule_id, slug, name, conditions=None, required=None, severity="medium"):
    return {
        "rule_id": rule_id,
        "rule_name": name,
        "detector_slug": slug,
        "finding_type": "opportunity",
        "scope": None,
        "required_fields": json.dumps(required or []),
        "conditions": json.dumps(conditions or {}),
        "severity": severity,
        "rule_status": 1,
        "version": 1,
    }


def _nick(rule_rows, policy_rows=None):
    return SimpleNamespace(
        policy_engine=PolicyEngine(policy_rows=policy_rows or []),
        rule_book=RuleBook(rule_rows=rule_rows),
        settings=SimpleNamespace(script_user="tester"),
        query_engine=None,
    )


def test_registry_holds_exactly_the_rules_in_the_rule_book():
    rows = [
        _rule(1, "price_variance_check", "Price Benchmark Variance"),
        _rule(2, "contract_expiry_check", "Contract Expiry Opportunity"),
    ]
    agent = OpportunityMinerAgent(_nick(rows))

    assert sorted(agent._get_policy_registry()) == [
        "contract_expiry_check",
        "price_variance_check",
    ]


def test_a_detector_with_no_rule_row_does_not_run():
    """Twelve handlers exist in code; only the ones with a rule row are live."""
    agent = OpportunityMinerAgent(_nick([_rule(1, "price_variance_check", "Price")]))

    registry = agent._get_policy_registry()

    assert "maverick_spend_check" not in registry
    assert "esg_opportunity_check" not in registry


def test_a_rule_naming_an_unknown_detector_is_skipped_loudly(caplog):
    rows = [
        _rule(1, "price_variance_check", "Price Benchmark Variance"),
        _rule(2, "no_such_detector", "Invented Rule"),
    ]
    agent = OpportunityMinerAgent(_nick(rows))

    with caplog.at_level("ERROR"):
        registry = agent._get_policy_registry()

    assert "no_such_detector" not in registry
    assert "no_such_detector" in caplog.text


def test_thresholds_come_from_the_rule_row():
    rows = [_rule(1, "contract_expiry_check", "Contract Expiry",
                  conditions={"negotiation_window_days": 45})]
    agent = OpportunityMinerAgent(_nick(rows))

    entry = agent._get_policy_registry()["contract_expiry_check"]

    assert entry["default_conditions"] == {"negotiation_window_days": 45}


def test_an_absent_threshold_is_not_a_zero():
    rows = [_rule(1, "maverick_spend_check", "Maverick Spend", conditions={})]
    agent = OpportunityMinerAgent(_nick(rows))

    entry = agent._get_policy_registry()["maverick_spend_check"]

    assert entry["default_conditions"] == {}


def test_required_fields_come_from_the_rule_row():
    rows = [_rule(1, "price_variance_check", "Price Benchmark Variance",
                  required=["supplier_id", "item_id"])]
    agent = OpportunityMinerAgent(_nick(rows))

    entry = agent._get_policy_registry()["price_variance_check"]

    assert entry["required_fields"] == ["supplier_id", "item_id"]


def test_rule_name_is_the_display_name():
    rows = [_rule(1, "price_variance_check", "Price Benchmark Variance")]
    agent = OpportunityMinerAgent(_nick(rows))

    entry = agent._get_policy_registry()["price_variance_check"]

    assert entry["policy_name"] == "Price Benchmark Variance"
    assert entry["policy_id"] == "price_variance_check"


def test_a_policy_row_cannot_reach_into_the_registry():
    """The separation guard.

    Detector configuration used to live in proc.bp_policy and was bound to
    detectors by fuzzy alias matching. That matching mutated each entry's alias
    set as it scanned, so one entry accumulated seventeen aliases -- including
    the bare policy ids and the linked-agent name shared by every opportunity
    policy -- and ended up bound to whichever policy happened to match last.
    Four of five policies collapsed onto a single detector.

    No policy row, however it is named, may alter a rule now.
    """
    policy_rows = [
        {
            "policy_id": 5,
            "policy_name": "PriceBenchmarkVariance",
            "policy_type": "opportunity",
            "policy_desc": "price_variance_check",
            "policy_details": json.dumps(
                {
                    "policy_identifier": "price_variance_check",
                    "default_conditions": {"variance_threshold_pct": 99.0},
                    "required_fields": ["hijacked"],
                }
            ),
        }
    ]
    rows = [_rule(1, "price_variance_check", "Price Benchmark Variance",
                  conditions={}, required=["supplier_id"])]
    agent = OpportunityMinerAgent(_nick(rows, policy_rows))

    entry = agent._get_policy_registry()["price_variance_check"]

    assert entry["default_conditions"] == {}
    assert entry["required_fields"] == ["supplier_id"]
    assert entry["policy_id"] == "price_variance_check"
    assert entry.get("source_policy") is None


def test_an_empty_rule_book_raises_rather_than_running_no_detectors():
    with pytest.raises(RuleBookUnavailable):
        RuleBook(rule_rows=[])


def test_a_missing_rule_book_is_an_outage_not_an_empty_registry():
    """An agent whose nick has no rule book must refuse, not detect nothing."""
    nick = SimpleNamespace(
        policy_engine=PolicyEngine(policy_rows=[]),
        rule_book=None,
        settings=SimpleNamespace(script_user="tester"),
        query_engine=None,
    )
    agent = OpportunityMinerAgent(nick)

    with pytest.raises(RuleBookUnavailable):
        agent._get_policy_registry()
