"""reason() as the orchestrator: route, build the context, run the loop, verify. The loop itself is faked."""
from types import SimpleNamespace

import pytest

from orchestration import agentnick_control as control
from services.tool_runtime import ToolCall, ToolRunResult

POLICY_ASK = "Is a 25,000 GBP purchase order allowed under our policy, and who is the approver?"
PLAIN_ASK = "What did we spend on packaging last year?"


@pytest.fixture
def captured(monkeypatch):
    got = {}

    def fake_run_tools(task, tools, system, **kw):
        got.update(task=task, tools=[t.name for t in tools], system=system, kw=kw)
        return ToolRunResult(model="m", system=system, tools_offered=[t.name for t in tools], answer="ok")

    monkeypatch.setattr(control, "run_tools", fake_run_tools)
    monkeypatch.delenv("AGENT_SKILL_ROUTER", raising=False)
    monkeypatch.delenv("AGENT_VERIFY", raising=False)
    return got


def PLAIN_SYSTEM():
    """The standing rules plus today's date: all a plain request is given (the date is the one fact no tool returns)."""
    import datetime as dt
    from orchestration import context_builder as cb
    return control._SYSTEM + "\n\n" + cb.today_line(dt.date.today())


def nick():
    return SimpleNamespace(agents={"opportunity_miner": SimpleNamespace()})


def test_a_request_a_skill_covers_gets_its_playbook_in_the_prompt_and_not_the_reading_tools(captured):
    res = control.reason(nick(), POLICY_ASK)
    assert "Skill loaded for this request: procurement-policy-compliance" in captured["system"]
    assert "list_skills" not in captured["tools"] and "use_skill" not in captured["tools"]
    assert "check_policy" in captured["tools"]
    assert [l["layer"] for l in res.context] == ["base", "today", "skill"]


def test_a_request_no_skill_covers_is_exactly_what_it_was_before(captured):
    res = control.reason(nick(), PLAIN_ASK)
    assert captured["system"] == PLAIN_SYSTEM()
    assert "list_skills" in captured["tools"] and "use_skill" in captured["tools"]
    assert [l["layer"] for l in res.context] == ["base", "today"]


def test_extra_instructions_are_kept_with_a_skill(captured):
    control.reason(nick(), POLICY_ASK, extra_system="Be brief.")
    s = captured["system"]
    assert s.index("Be brief.") < s.index("Skill loaded for this request")


def test_a_skill_whose_tools_the_agent_was_not_given_is_not_injected(captured):
    res = control.reason(nick(), POLICY_ASK, only=["get_policy", "list_governance"])
    assert "Skill loaded" not in captured["system"] and captured["system"] == PLAIN_SYSTEM()
    assert res.context[0]["layer"] == "base"


def test_the_router_can_be_switched_off(captured, monkeypatch):
    monkeypatch.setenv("AGENT_SKILL_ROUTER", "off")
    control.reason(nick(), POLICY_ASK)
    assert "Skill loaded" not in captured["system"] and "use_skill" in captured["tools"]


def test_a_router_failure_never_breaks_the_run(captured, monkeypatch):
    from orchestration import skill_router
    monkeypatch.setattr(skill_router, "load_entries", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = control.reason(nick(), POLICY_ASK)
    assert captured["system"] == PLAIN_SYSTEM() and res.answer == "ok"


def test_the_verify_step_is_handed_to_the_loop_and_can_be_switched_off(captured, monkeypatch):
    control.reason(nick(), PLAIN_ASK)
    assert callable(captured["kw"]["verify"])
    monkeypatch.setenv("AGENT_VERIFY", "off")
    control.reason(nick(), PLAIN_ASK)
    assert captured["kw"]["verify"] is None


def test_the_policy_skills_check_policy_reply_becomes_the_answer(captured):
    control.reason(nick(), POLICY_ASK)
    verify = captured["kw"]["verify"]
    result = SimpleNamespace(calls=[ToolCall(name="check_policy", arguments={}, ok=True,
                                             result={"reply_text": "Compliant with conditions."})])
    v = verify("model wording with 99,999", result, False)
    assert v.action == "replace" and v.answer == "Compliant with conditions."


def test_without_a_skill_a_figure_from_nowhere_sends_the_model_back(captured):
    control.reason(nick(), PLAIN_ASK)
    v = captured["kw"]["verify"]("We spent 123,456 GBP.", SimpleNamespace(calls=[]), False)
    assert v.action == "retry" and "123,456" in v.untraced


def test_what_the_orchestrator_did_is_logged_on_the_result_even_when_the_loop_is_faked(captured):
    res = control.reason(nick(), POLICY_ASK)
    assert res.context and all({"layer", "chars", "source"} <= set(l) for l in res.context)


def test_the_argument_check_is_handed_to_the_loop_and_refuses_an_invented_rate(captured):
    control.reason(nick(), "Compare two bids, quality weight 0.6, price weight 0.4.")
    check = captured["kw"]["check_arguments"]
    assert callable(check)
    r = check("run_calculation", {"inputs": {"cost_of_capital": 0.08}}, SimpleNamespace(calls=[]))
    assert r and r["reasonCode"] == "figure_not_sourced"
    assert check("run_calculation", {"inputs": {"criteria": [{"weight": "0.6"}]}}, SimpleNamespace(calls=[])) is None


def test_the_argument_check_uses_the_request_and_can_be_switched_off(captured, monkeypatch):
    control.reason(nick(), "Our cost of capital is 8 percent.")
    assert captured["kw"]["check_arguments"]("run_calculation", {"inputs": {"cost_of_capital": 0.08}},
                                              SimpleNamespace(calls=[])) is None
    monkeypatch.setenv("AGENT_VERIFY", "off")
    control.reason(nick(), PLAIN_ASK)
    assert captured["kw"]["check_arguments"] is None


def test_a_request_a_skill_covers_is_offered_only_the_skills_own_tools(captured):
    from services import skills as SK
    from orchestration import context_builder as cb
    res = control.reason(nick(), POLICY_ASK)
    needed = cb.tools_needed(SK.load_skill("procurement-policy-compliance").body)
    offered = set(captured["tools"])
    assert offered and offered <= needed, (offered - needed)
    assert "list_skills" not in offered and "use_skill" not in offered
    assert "check_policy" in offered


def test_the_tool_scope_can_be_switched_off(captured, monkeypatch):
    monkeypatch.setenv("AGENT_SKILL_TOOL_SCOPE", "off")
    control.reason(nick(), POLICY_ASK)
    assert "get_corpus_facts" in captured["tools"]
