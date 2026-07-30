import json

from src.agents.base_agent import AgentContext, AgentStatus


def _ctx(data):
    return AgentContext(workflow_id="W1", agent_id="negotiation_advisor",
                        user_id="buyer", input_data=data)


_ADVICE = {"advice_id": "A-1", "deal_id": "D-1", "quadrant": "Leverage",
           "style": "Competitive", "plays": [{"lever": "Commercial",
                                              "state": "ready"}]}


def _agent(monkeypatch, advice=_ADVICE, turn=None):
    import src.agents.negotiation_advisor_agent as mod
    agent = mod.NegotiationAdvisorAgent.__new__(mod.NegotiationAdvisorAgent)
    agent._with_plan = lambda ctx, out: out
    monkeypatch.setattr(mod, "build_advice", lambda deal_id, **kw: advice)
    monkeypatch.setattr(mod, "apply_turn",
                        lambda deal_id, message, **kw: turn or advice)
    return agent


def test_run_returns_advice_for_a_deal(monkeypatch):
    agent = _agent(monkeypatch)
    out = agent.run(_ctx({"deal_id": "D-1"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["quadrant"] == "Leverage"
    assert out.data["plays"]


def test_run_applies_a_turn_when_an_action_is_given(monkeypatch):
    agent = _agent(monkeypatch, turn=dict(_ADVICE, style="Principled"))
    out = agent.run(_ctx({"deal_id": "D-1", "action": "override",
                          "style": "Principled"}))
    assert out.data["style"] == "Principled"


def test_missing_deal_id_is_reported_not_raised(monkeypatch):
    agent = _agent(monkeypatch)
    out = agent.run(_ctx({}))
    assert out.status == AgentStatus.SUCCESS
    assert out.error


def test_unknown_deal_is_reported_not_raised(monkeypatch):
    agent = _agent(monkeypatch, advice=None)
    out = agent.run(_ctx({"deal_id": "NOPE"}))
    assert out.error


def test_run_signature_matches_the_shared_contract():
    import inspect
    from src.agents.negotiation_advisor_agent import NegotiationAdvisorAgent
    params = list(inspect.signature(NegotiationAdvisorAgent.run).parameters)
    assert params == ["self", "context"]


def test_registered_in_the_catalogue():
    with open("agent_definitions.json", encoding="utf-8") as fh:
        agents = json.load(fh)["agents"]
    entry = next(a for a in agents if a["slug"] == "negotiation_advisor")
    assert entry["class_path"] == (
        "agents.negotiation_advisor_agent.NegotiationAdvisorAgent")
    assert len({a["slug"] for a in agents}) == len(agents)
    assert len({a["agentId"] for a in agents}) == len(agents)
