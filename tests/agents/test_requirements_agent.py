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


def test_asking_for_a_scope_returns_a_scope_not_a_question(monkeypatch):
    """The reported defect: "provide a list of requirements" got another question.

    Asking for a scope must produce the scope, with ONE confirm question after
    it — never an interrogation.
    """
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[
            # First turn carries a brief, so the facts in it are extracted before
            # the scope is drafted (see test_scope_request_with_a_brief_...).
            {"updates": {"title": "Managed cloud data platform"}, "next_question": ""},
            {"areas": [
                {"area": "Data protection & residency",
                 "requirement": "All data remains in UK regions; sub-processors disclosed."},
            ]},
        ],
        redis=_FakeRedis(),
    )
    # First turn for this session id (the UI passes its requirement id): no prior row.
    monkeypatch.setattr(ra_mod.requirement_service, "get_by_session", lambda sid: None)
    out = agent.run(_ctx({
        "session_id": "REQ-2041",
        "category": "SaaS / IT",
        "brief": "Managed cloud data platform — consolidate three data tools onto one platform.",
        "message": "tell me the requirements I should have for a managed cloud platform",
    }))

    assert out.status == AgentStatus.SUCCESS
    assert out.data["mode"] == "proposed_scope"
    scope = out.data["scope"]
    assert len(scope["areas"]) >= 8               # a scope, not one question
    assert scope["family"] == "saas_it"           # commodity-aware
    # The model's tailored wording is used and labelled as such.
    residency = next(a for a in scope["areas"] if a["area"] == "Data protection & residency")
    assert residency["requirement"] == "All data remains in UK regions; sub-processors disclosed."
    assert residency["source"] == "tailored"
    # Exactly one follow-up question, and it asks for confirmation.
    assert out.data["next_question"].count("?") == 1
    assert "confirm" in out.data["next_question"].lower() or \
           "right" in out.data["next_question"].lower()


def test_scope_request_with_a_brief_still_captures_the_brief_facts(monkeypatch):
    """A first-turn brief carries the title, value and term. Propose mode must
    extract those before drafting, or the scope is written about a category label
    ("Supplier delivers SaaS / IT") instead of the buyer's actual need."""
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[
            # 1) extraction from the brief, 2) scope tailoring
            {"updates": {"title": "Managed cloud data platform", "category": "SaaS / IT"},
             "next_question": "How many users?"},
            {"areas": []},
        ],
        redis=_FakeRedis(),
    )
    out = agent.run(_ctx({
        "brief": "Managed cloud data platform — consolidate three data tools.",
        "message": "provide a list of requirements",
    }))
    assert out.data["mode"] == "proposed_scope"
    assert out.data["requirement"]["title"] == "Managed cloud data platform"
    # The extraction call's question is discarded — we proposed instead of asking it.
    assert out.data["next_question"] != "How many users?"
    blob = " ".join(a["requirement"] for a in out.data["scope"]["areas"])
    assert "Managed cloud data platform" in blob


def test_proposed_scope_is_never_persisted_as_buyer_stated_fact(monkeypatch):
    """A proposal is advice, not data. It must not fill requirement fields."""
    persisted = []
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"areas": [{"area": "Volume & scale",
                                  "requirement": "Supports 500 users at peak."}]}],
        redis=_FakeRedis(),
    )
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"category": "SaaS / IT",
                          "message": "what requirements should I have for a cloud platform?"}))

    # No field was invented from the proposal — quantity/date/location stay missing.
    for field in ("quantity", "needed_by_date", "delivery_location"):
        assert field in out.data["missing_fields"]
        assert out.data["requirement"].get(field) is None
    assert persisted[-1]["status"] == "gathering"
    assert persisted[-1].get("specifications") is None   # nothing adopted yet


def test_scope_survives_the_llm_being_unreachable(monkeypatch):
    """The buyer asked for a scope; an Ollama outage must not turn that into a question."""
    agent = _make_agent(monkeypatch, llm_payloads=[], redis=_FakeRedis())

    def _boom(**kw):
        raise ConnectionError("ollama is not reachable")

    agent.call_ollama = _boom
    out = agent.run(_ctx({"category": "Works & construction",
                          "message": "give me a scope of requirements"}))

    assert out.data["mode"] == "proposed_scope"
    assert len(out.data["scope"]["areas"]) >= 8
    assert out.data["scope"]["basis"] == "template"    # honest: generic, not tailored
    assert all(a["source"] == "template" for a in out.data["scope"]["areas"])
    assert out.data["next_question"]                    # still ends with one confirm question


def test_accepting_a_proposal_adopts_it_into_specifications(monkeypatch):
    redis = _FakeRedis()
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"areas": []}, {"updates": {}, "next_question": "How many users?"}],
        redis=redis,
    )
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    first = agent.run(_ctx({"category": "SaaS / IT", "message": "provide a list of requirements"}))
    session_id = first.data["session_id"]
    second = agent.run(_ctx({"session_id": session_id, "message": "yes, looks good — use that"}))

    assert second.data["mode"] == "scope_accepted"
    specs = persisted[-1]["specifications"]
    assert specs["source"] == "agent_proposed"          # labelled, not passed off as buyer-authored
    assert len(specs["scope_areas"]) >= 8
    # Adoption does not fake completeness: the real gaps are still asked about.
    assert second.data["next_question"]
    assert second.data["complete"] is False


def test_answering_a_question_still_elicits(monkeypatch):
    """Guard the other direction: an answer containing "scope" must not be
    mistaken for a request for advice, or the buyer's facts get dropped."""
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Cloud data platform", "category": "SaaS / IT"},
                       "next_question": "How many users at peak?"}],
        redis=_FakeRedis(),
    )
    out = agent.run(_ctx({
        "message": "Migration is in scope; bespoke dashboards stay in-house, so out of scope.",
    }))
    assert out.data["mode"] == "elicitation"
    assert out.data["requirement"]["title"] == "Cloud data platform"
    assert out.data["next_question"] == "How many users at peak?"
    assert "scope" not in out.data


def test_completed_requirement_also_returns_the_scope(monkeypatch):
    """When gathering finishes, the buyer gets a supplier-ready scope — not just
    a 'captured' message that leaves them to write the requirement themselves."""
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[
            {"updates": {"title": "Laptops", "category": "IT", "quantity": 10,
                         "needed_by_date": "2026-07-01", "delivery_location": "London HQ"},
             "next_question": ""},
            {"areas": [{"area": "Licensing model", "requirement": "Perpetual, 10 seats."}]},
        ],
        redis=_FakeRedis(),
    )
    out = agent.run(_ctx({"message": "10 laptops to London HQ by July 1"}))
    assert out.data["complete"] is True
    assert out.data["mode"] == "complete"
    assert len(out.data["scope"]["areas"]) >= 8


def test_untyped_field_value_is_rejected_not_persisted(monkeypatch):
    """Found live: the model answered needed_by_date with "3 years from contract
    start". bp_requirement.needed_by_date is a DATE, so the insert raised and the
    whole turn — scope included — was lost. The value must be dropped and named."""
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Cloud platform",
                                   "needed_by_date": "3 years from contract start"},
                       "next_question": "When do you need it?"}],
        redis=_FakeRedis(),
    )
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"message": "we need a cloud platform for three years"}))

    assert out.status == AgentStatus.SUCCESS
    assert out.data["requirement"]["title"] == "Cloud platform"      # good field kept
    assert out.data["requirement"].get("needed_by_date") is None     # bad field dropped
    assert "needed_by_date" in out.data["missing_fields"]            # so it gets asked for
    assert out.data["rejected_fields"][0]["field"] == "needed_by_date"
    assert persisted[-1].get("needed_by_date") is None


def test_persist_failure_does_not_lose_the_answer(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"areas": []}],
        redis=_FakeRedis(),
    )

    def _db_down(rec):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(ra_mod.requirement_service, "persist", _db_down)
    out = agent.run(_ctx({"category": "SaaS / IT", "message": "give me a scope"}))

    assert out.status == AgentStatus.SUCCESS
    assert out.data["persisted"] is False          # reported, not hidden
    assert len(out.data["scope"]["areas"]) >= 8    # the answer still arrives


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
