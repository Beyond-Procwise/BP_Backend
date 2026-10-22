"""get_policy: one call per topic. A miss is terminal unless a listed name clearly fits."""
from types import SimpleNamespace

import pytest

from orchestration import agentnick_control as control
from services.governance_tools import tools as GT

CATALOGUE = {"policies": [{"slug": "approval_threshold_policy"}, {"slug": "receipt_tolerance_policy"},
                          {"slug": "supplier_scorecard"}]}


@pytest.fixture
def get_policy(monkeypatch):
    monkeypatch.setattr(GT, "get_policy", lambda q: {"slug": q, "rules": {}} if q == "approval_threshold_policy" else {})
    monkeypatch.setattr(GT, "list_governance", lambda agent=None: CATALOGUE)
    tools = {t.name: t for t in control.build_tools(SimpleNamespace(agents={"opportunity_miner": SimpleNamespace()}))}
    return tools["get_policy"]


def test_slug_type_and_agent_arguments_are_accepted_as_the_query(get_policy):
    assert get_policy.handler(slug="approval_threshold_policy")["slug"] == "approval_threshold_policy"
    assert get_policy.handler(type="approval_threshold_policy")["slug"] == "approval_threshold_policy"
    assert get_policy.handler(query="nothing", extra="ignored")["found"] is False
    assert set(get_policy.parameters["properties"]) >= {"query", "slug", "type", "agent"}
    assert "required" not in get_policy.parameters


def test_a_miss_says_the_list_is_complete_and_the_topic_is_ungoverned(get_policy):
    out = get_policy.handler(query="buying-channel")
    assert out["found"] is False
    assert "complete list" in out["note"] and "not governed" in out["note"]
    assert "Retry" not in out["note"] and "retry" not in out["note"]
    assert out["available"] == ["approval_threshold_policy", "receipt_tolerance_policy", "supplier_scorecard"]


def test_a_miss_suggests_only_the_names_that_share_a_word_with_the_query(get_policy):
    assert get_policy.handler(query="approval bands purchase order")["suggested"] == ["approval_threshold_policy"]
    assert get_policy.handler(query="SC-007")["suggested"] == []


def test_an_empty_call_is_told_what_the_tool_needs(get_policy):
    out = get_policy.handler()
    assert out["found"] is False and "needs a query" in out["note"]


def test_the_description_says_one_call_per_topic(get_policy):
    assert "One call per topic" in get_policy.description and "retry with one of those" not in get_policy.description
