import src.agents.requirements_agent as ra_mod
from src.agents.requirements_agent import RequirementsAgent
from src.agents.base_agent import AgentContext, AgentStatus


class _FakeRedis:
    def __init__(self):
        self.store = {}
    def set(self, k, v):
        self.store[k] = v
    def get(self, k):
        return self.store.get(k)


def _make_agent(monkeypatch, llm_payloads, redis=None):
    """llm_payloads: list of dicts returned (as JSON) by successive call_ollama calls."""
    agent = RequirementsAgent.__new__(RequirementsAgent)  # bypass heavy __init__
    agent._workflow_context = None
    calls = {"i": 0}

    def fake_call_ollama(prompt=None, model=None, format=None, messages=None, **kw):
        payload = llm_payloads[calls["i"]]
        calls["i"] += 1
        import json as _json
        return {"response": _json.dumps(payload)}

    agent.call_ollama = fake_call_ollama
    agent.resolve_prompt = lambda name, **fmt: None
    agent.governing_policy = lambda name: None
    agent._with_plan = lambda ctx, out: out  # bypass BaseAgent plan internals
    agent.emit_signal = lambda st, msg, data=None: None
    monkeypatch.setattr(ra_mod, "get_redis_client", lambda: redis)
    monkeypatch.setattr(ra_mod.requirement_service, "seed_context", lambda c: {})
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: None)
    return agent


def _ctx(data):
    return AgentContext(workflow_id="W1", agent_id="requirements", user_id="alice", input_data=data)


def test_incomplete_turn_asks_next_question(monkeypatch):
    redis = _FakeRedis()
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Laptops", "category": "IT"},
                       "next_question": "How many laptops do you need?"}],
        redis=redis,
    )
    out = agent.run(_ctx({"message": "I need laptops for the IT team", "created_by": "alice"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False
    assert out.data["next_question"] == "How many laptops do you need?"
    assert "quantity" in out.data["missing_fields"]
    assert out.data["session_id"] in [k.split(":")[1] for k in redis.store]


def test_complete_turn_persists_and_emits_signals(monkeypatch):
    redis = _FakeRedis()
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {
            "title": "Laptops", "category": "IT", "quantity": 10,
            "needed_by_date": "2026-07-01", "delivery_location": "London HQ"},
            "next_question": ""}],
        redis=redis,
    )
    emitted = []
    agent.emit_signal = lambda st, msg, data=None: emitted.append((st, data))
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"message": "10 laptops to London HQ by July 1", "created_by": "alice"}))
    assert out.data["complete"] is True
    assert out.data["completeness_score"] == 1.0
    assert persisted and persisted[0]["status"] == "complete"
    signal_targets = [d.get("agent") for _, d in emitted]
    assert "supplier_ranking" in signal_targets
    assert "email_drafting" in signal_targets


def test_brief_is_parsed_on_first_turn(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Office chairs", "category": "Furniture"},
                       "next_question": "What quantity?"}],
        redis=_FakeRedis(),
    )
    out = agent.run(_ctx({"brief": "We urgently need office chairs for the new floor."}))
    assert out.data["requirement"]["title"] == "Office chairs"
    assert out.data["complete"] is False


def test_malformed_llm_json_does_not_crash(monkeypatch):
    agent = _make_agent(monkeypatch, llm_payloads=[], redis=_FakeRedis())
    agent.call_ollama = lambda **kw: {"response": "not json at all"}
    out = agent.run(_ctx({"message": "hello"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False  # nothing filled, still gathering
