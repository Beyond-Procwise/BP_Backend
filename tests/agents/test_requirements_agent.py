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
    calls = {"i": 0, "kwargs": []}

    class _GenResponse:
        """Mirror ollama's GenerateResponse: attribute access, NOT a dict."""
        def __init__(self, response):
            self.response = response

    def fake_call_ollama(prompt=None, model=None, format=None, messages=None, **kw):
        payload = llm_payloads[calls["i"]]
        calls["i"] += 1
        calls["kwargs"].append(kw)
        import json as _json
        return _GenResponse(_json.dumps(payload))

    agent.call_ollama = fake_call_ollama
    agent._llm_calls = calls
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
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))
    out = agent.run(_ctx({"message": "I need laptops for the IT team", "created_by": "alice"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False
    assert out.data["next_question"] == "How many laptops do you need?"
    assert "quantity" in out.data["missing_fields"]
    assert out.data["session_id"] in [k.split(":")[1] for k in redis.store]
    # The gathering row is persisted every turn (durable session backing store).
    assert persisted and persisted[-1]["status"] == "gathering"
    # think=False is required for the AgentNick reasoning model to return output.
    assert agent._llm_calls["kwargs"][0].get("think") is False


def test_reloads_prior_state_from_db_when_redis_absent(monkeypatch):
    # Redis disabled (get_redis_client -> None). Turn 2 must reload the gathering
    # row by session_id from the DB and continue, not start a fresh requirement.
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"quantity": 25, "needed_by_date": "2026-08-15",
                                   "delivery_location": "London HQ"},
                       "next_question": ""}],
        redis=None,
    )
    prior_row = {
        "requirement_id": "REQ-prior", "created_by": "alice", "status": "gathering",
        "title": "Chairs", "category": "Furniture", "seed_context": {},
    }
    monkeypatch.setattr(ra_mod.requirement_service, "get_by_session",
                        lambda sid: prior_row if sid == "S-prior" else None)
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"session_id": "S-prior",
                          "message": "25 units to London HQ by 2026-08-15"}))
    assert out.data["requirement"]["title"] == "Chairs"        # reloaded from DB
    assert out.data["requirement"]["category"] == "Furniture"  # reloaded from DB
    assert out.data["complete"] is True                        # + new fields → done
    assert out.data["requirement_id"] == "REQ-prior"           # continuity, not a new id
    assert persisted[-1]["status"] == "complete"


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


def test_db_template_with_literal_json_braces_does_not_crash(monkeypatch):
    # Regression: the DB-seeded elicitation prompt contains literal JSON braces
    # ({"updates": {}, ...}). str.format() would raise KeyError on them; the
    # agent must use plain substitution and survive.
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Chairs"}, "next_question": "How many?"}],
        redis=_FakeRedis(),
    )
    db_template = (
        "Current requirement: {requirement}\nStill missing: {missing}\n"
        "Buyer message: {message}\n"
        'Respond ONLY with JSON: {"updates": {}, "next_question": ""}'
    )
    agent.resolve_prompt = lambda name, **fmt: db_template
    out = agent.run(_ctx({"message": "I need chairs"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["requirement"]["title"] == "Chairs"
    assert out.data["next_question"] == "How many?"


def test_malformed_llm_json_does_not_crash(monkeypatch):
    agent = _make_agent(monkeypatch, llm_payloads=[], redis=_FakeRedis())
    agent.call_ollama = lambda **kw: {"response": "not json at all"}
    out = agent.run(_ctx({"message": "hello"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False  # nothing filled, still gathering


def test_llm_transport_failure_keeps_the_turn_alive(monkeypatch):
    # Regression: `raw`/`result` were bound only inside the try block, but the
    # "no field updates" warning after it read them. So any transport failure
    # (Ollama down, timeout, model not loaded) raised UnboundLocalError instead
    # of logging, and the whole turn came back FAILED. The turn must survive and
    # keep asking, so the buyer's session is not destroyed by one LLM blip.
    agent = _make_agent(monkeypatch, llm_payloads=[], redis=_FakeRedis())

    def _boom(**kw):
        raise ConnectionError("ollama is not reachable")

    agent.call_ollama = _boom
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"message": "I need 40 monitors", "created_by": "alice"}))

    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False
    assert out.data["next_question"] == ""
    assert "title" in out.data["missing_fields"]
    # The durable gathering row is still written, so the next turn can resume.
    assert persisted and persisted[-1]["status"] == "gathering"
