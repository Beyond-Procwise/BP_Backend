import pytest

from src.orchestration import agentnick_control as control
from services.governance_tools import tools as GT

APPROVAL = {"slug": "approval_threshold_policy", "policy_name": "ApprovalThresholdPolicy",
            "details": {"rules": {"currency": "GBP", "default_threshold_gbp": 10000}, "required_role": "Approver"}}


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setattr(GT, "active_policy_count", lambda: 31)
    monkeypatch.setattr(GT, "refresh", lambda strict=False: None)
    monkeypatch.setattr(GT, "list_governance", lambda agent=None: {"policies": [{"slug": "approval_threshold_policy"}]})
    monkeypatch.setattr(GT, "get_policy", lambda q: APPROVAL if q == "approval_threshold_policy" else {})
    return {t.name: t for t in control._policy_check_tools()}["check_policy"]


def test_the_tool_is_in_the_agents_tool_set():
    assert "check_policy" in [t.name for t in control.build_tools(None)]


def test_it_reads_the_policy_itself_and_returns_a_verdict(tool):
    v = tool.handler(total_commitment=25000, currency="GBP")
    assert v["verdict"] == "COMPLIANT_WITH_CONDITIONS" and v["policy"]["threshold"] == 10000
    assert "Approver" in v["reply_text"]


def test_zero_active_policies_is_an_outage_not_an_empty_rulebook(tool, monkeypatch):
    monkeypatch.setattr(GT, "active_policy_count", lambda: 0)
    assert tool.handler(total_commitment=1, currency="GBP")["verdict"] == "INSUFFICIENT_INFO"


def test_a_failing_policy_read_fails_closed(tool, monkeypatch):
    def boom(q): raise RuntimeError("db down")
    monkeypatch.setattr(GT, "get_policy", boom)
    v = tool.handler(total_commitment=1, currency="GBP")
    assert v["verdict"] == "INSUFFICIENT_INFO"


def test_junk_arguments_do_not_crash_it(tool):
    assert tool.handler()["verdict"] == "INSUFFICIENT_INFO"
    assert tool.handler(total_commitment="x", orders="nope")["verdict"] == "INSUFFICIENT_INFO"


def test_the_description_tells_the_model_to_relay_the_reply_unchanged():
    d = {t.name: t for t in control._policy_check_tools()}["check_policy"].description.lower()
    assert "reply_text" in d and "do not" in d


def test_reading_the_approval_policy_points_the_model_at_check_policy(monkeypatch):
    monkeypatch.setattr(GT, "get_policy", lambda q: dict(APPROVAL) if "approval" in q else {})
    monkeypatch.setattr(GT, "list_governance", lambda agent=None: {"policies": []})
    get = {t.name: t for t in control._governance_tools()}["get_policy"]
    out = get.handler(query="approval_threshold_policy")
    assert "check_policy" in out["note"] and out["slug"] == "approval_threshold_policy"


def test_other_policies_carry_no_such_note(monkeypatch):
    monkeypatch.setattr(GT, "get_policy", lambda q: {"slug": "negotiation_bounds_policy", "details": {}})
    monkeypatch.setattr(GT, "list_governance", lambda agent=None: {"policies": []})
    get = {t.name: t for t in control._governance_tools()}["get_policy"]
    assert "note" not in get.handler(query="negotiation_bounds_policy")
