"""The three drafting paths with the stages live, against a fake model that is good, then bad."""

import json
from types import SimpleNamespace

from tests.services.test_draft_assurance import FakeConn, TABLES, _family_rules, _free_prompt_rules
from tests.services.test_draft_run import FAMILY_V2, TONE_SQL, _policies
from tests.services.test_draft_capture import Conn as CapConn, Cur, _row
from src.services.draft_assurance import capture
from pathlib import Path

REQ = "Please ask Acme to confirm the price on PO-77123 and reply within the week"
CLS = {"family_id": "free_prompt", "confidence": 0.9,
       "candidates": [{"family_id": "free_prompt", "confidence": 0.9}, {"family_id": "negotiation_counter", "confidence": 0.1}],
       "lookup_keys": {"po_number": "PO-77123"}, "user_instruction": "ask Acme to confirm the price"}
PLAN = {"goal": "Get the price confirmed", "key_points": ["Confirm the price on PO-77123"], "explicit_ask": "Confirm the price",
        "deadline": "within the week", "tone_rationale": "Courteous", "risks_to_avoid": [], "reasoned": {}, "assumptions": []}
JUDGE = {"scores": {"completeness": 4}, "rationale": "fine"}
PROMPTS = {"email_family_classify": "C {families}", "email_brief_plan": "P {facts}", "email_draft_judge": "J {rubric}"}


def _agent(monkeypatch, model):
    """An agent whose stage model is ``model(system, user) -> str`` and whose composer says one fixed thing."""
    from agents import email_drafting_agent as module
    from src.services import supplier_contact
    agent = module.EmailDraftingAgent()
    composed = []

    def chat(m, system, user, **k):
        composed.append(user)
        return "Subject: Price\nPlease confirm the price on PO-77123 within the week?"

    monkeypatch.setattr(module, "_chat", chat)
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20260101")
    monkeypatch.setattr(agent, "_master_contact", lambda sid: supplier_contact.SupplierContact(emails=["a@x.test"], name="Alex"))
    monkeypatch.setattr(agent, "resolve_prompt", lambda name, **k: PROMPTS.get(name))
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: {"message": {"content": model(kw["messages"][0]["content"], kw["messages"][1]["content"])}})
    agent.agent_nick.get_db_connection = lambda: FakeConn(TABLES)
    agent.agent_nick.policy_engine = _policies()
    agent._composed = composed
    return agent


def good_model(system, user):
    return json.dumps(CLS if system.startswith("C ") else PLAN if system.startswith("P ") else JUDGE)


def _prompt(agent, **extra):
    return agent.from_prompt(REQ, context={"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"], **extra})


def test_from_prompt_is_classified_planned_written_from_the_brief_and_judged(monkeypatch):
    agent = _agent(monkeypatch, good_model)
    a = _prompt(agent)["assurance"]
    assert a["family_source"] == "classified" and a["family_id"] == "free_prompt"
    assert a["classification"]["lookup_keys"] == {"po_number": "PO-77123"}
    assert a["brief"]["goal"] == "Get the price confirmed"
    assert a["judge"]["status"] == "scored"
    assert a["tone"]["variables"]["escalation_level"] == 1                 # no prior-contact data -> declared default
    assert a["exemplars"] == {"ids": [], "scope": "none"}
    assert '"brief"' in agent._composed[0] and "Get the price confirmed" in agent._composed[0]   # plan, then draft


def test_from_prompt_with_a_model_that_returns_garbage_at_every_stage_still_drafts_and_records_why(monkeypatch):
    agent = _agent(monkeypatch, lambda system, user: "I'm sorry, I cannot do that {")
    draft = _prompt(agent)
    a = draft["assurance"]
    assert "Please confirm the price" in draft["text"]                      # the draft still exists
    st = {k: v["status"] for k, v in a["stage_status"].items()}
    assert (st["classify"], st["brief"], st["judge"]) == ("invalid", "invalid", "invalid")
    assert a["family_source"] == "fallback" and a["family_id"] == "free_prompt" and a["judge"]["status"] == "invalid"
    assert '"brief"' not in agent._composed[0]                               # a refused plan is never fed to the writer


def test_a_model_that_invents_a_family_does_not_get_it_used(monkeypatch):
    bad = {**CLS, "family_id": "wire_transfer_request"}
    agent = _agent(monkeypatch, lambda s, u: json.dumps(bad if s.startswith("C ") else PLAN if s.startswith("P ") else JUDGE))
    a = _prompt(agent)["assurance"]
    assert a["family_id"] == "free_prompt" and a["family_source"] == "fallback"
    assert "not a configured family" in a["stage_status"]["classify"]["reason"]


def test_the_writer_inventing_a_po_number_is_caught_even_with_good_stages(monkeypatch):
    from agents import email_drafting_agent as module
    agent = _agent(monkeypatch, good_model)
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Price\nPlease confirm PO-99999 within the week?")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: None)
    kinds = {v["kind"] for v in _prompt(agent)["assurance"]["violations"]}
    assert "ungrounded_reference" in kinds


def test_the_writer_stating_a_figure_in_no_fact_is_caught(monkeypatch):
    from agents import email_drafting_agent as module
    agent = _agent(monkeypatch, good_model)
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Price\nWe will pay 61.25 GBP. Please confirm within the week?")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: None)
    assert any(v["kind"] == "ungrounded_figure" for v in _prompt(agent)["assurance"]["violations"])


def test_a_low_confidence_classification_leaves_the_draft_not_ready(monkeypatch):
    low = {**CLS, "confidence": 0.4, "candidates": [{"family_id": "free_prompt", "confidence": 0.4}, {"family_id": "negotiation_counter", "confidence": 0.35}]}
    agent = _agent(monkeypatch, lambda s, u: json.dumps(low if s.startswith("C ") else PLAN if s.startswith("P ") else JUDGE))
    a = _prompt(agent)["assurance"]
    assert a["clarification"]["options"] == ["free_prompt", "negotiation_counter"] and a["ready"] is False


# --- the declared paths -------------------------------------------------------------------------------------

DECISION = {"supplier_id": "S-1", "supplier_name": "Acme", "workflow_id": "wf-1", "to": "a@x.test",
            "current_offer": 49.0, "currency": "GBP", "counter_price": 44.8, "response_deadline": "30 October 2026",
            "asks": ["Confirm revised pricing"], "rationale": "Anchor low",
            "reasoned_basis": {"response_deadline": ["email_thread_summary"]}}


def _decision_agent(monkeypatch):
    from agents import email_drafting_agent as module
    agent = _agent(monkeypatch, lambda s, u: json.dumps({"scores": {"ask_is_specific": 5, "position_follows_from_offer": 4,
                                                                      "deadline_stated": 5, "tone_matches_escalation_level": 4, "concise": 5}}))
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Re\nThank you for your offer of 47.50 GBP. We propose 44.80 GBP. Please confirm by 30 October 2026?")
    return agent


def test_from_decision_declares_its_family_and_still_runs_every_check(monkeypatch):
    a = _decision_agent(monkeypatch).from_decision(dict(DECISION))["assurance"]
    assert a["family_source"] == "declared" and a["stage_status"]["classify"]["status"] == "not_run"
    assert a["facts"]["supplier_current_offer"]["row_id"] == "2" and a["conflicts"][0]["resolution"] == "postgres_wins"
    assert a["judge"]["status"] == "scored" and a["tone"] and a["brief"]["goal"] == "Anchor low"
    assert a["accountability"] == {"initiated_by": "EmailDraftingAgent", "kind": "agent"}


def test_the_counter_path_records_the_same_stages(monkeypatch):
    from agents.base_agent import AgentContext
    agent = _decision_agent(monkeypatch)
    stored = []
    monkeypatch.setattr(agent, "_store_draft", lambda d: stored.append(d))
    monkeypatch.setattr(agent, "_record_learning_events", lambda *a, **k: None)
    payload = {**DECISION, "recipients": ["a@x.test"]}
    ctx = AgentContext(workflow_id="wf-1", agent_id="email_drafting", user_id="nick@acme.test", input_data=payload)
    agent._handle_negotiation_counter(ctx, payload)
    a = stored[0]["assurance"]
    assert a["accountability"] == {"initiated_by": "nick@acme.test", "kind": "user"}
    assert set(a["stage_status"]) == {"classify", "tone", "exemplars", "brief", "judge", "authority", "steering"}


def test_the_whole_record_fits_the_capture_row(monkeypatch):
    """Every stage output is accepted by the INSERT: the column list and the parameter list agree."""
    a = _decision_agent(monkeypatch).from_decision(dict(DECISION))
    cur = Cur()
    assert capture.record_draft(CapConn(cur), {**a, "unique_id": a["unique_id"], "assurance": a["assurance"]}) == 99
    row = _row(cur)
    assert json.loads(row["stage_status"])["judge"]["status"] == "captured"
    assert row["exemplar_ids"] == "[]" and row["family_source"] == "declared" and row["initiated_by"] == "EmailDraftingAgent"
    assert json.loads(row["tone_sources"])["escalation_level"]["source"] in ("postgres", "default")


# --- the repair pass must really repair ------------------------------------------------------------------

def _broken_writer(agent, monkeypatch):
    from agents import email_drafting_agent as module
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Price\nWe will pay 61.25 GBP. Please confirm within the week?")


def test_a_refusal_is_not_accepted_as_a_repair(monkeypatch):
    agent = _agent(monkeypatch, good_model)
    _broken_writer(agent, monkeypatch)
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: "I'm sorry, I cannot help with that request, please try again later today")
    draft = _prompt(agent)
    assert "61.25" in draft["text"]                                       # the original stays; nothing was destroyed
    a = draft["assurance"]
    assert a["repaired"] is False and "different email" in a["repair_rejected"]
    assert any(v["kind"] == "ungrounded_figure" for v in a["violations"])


def test_a_repair_that_still_fails_is_not_accepted(monkeypatch):
    agent = _agent(monkeypatch, good_model)
    _broken_writer(agent, monkeypatch)
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: b.replace("61.25", "62.50"))
    a = _prompt(agent)["assurance"]
    assert a["repaired"] is False and "adds a new problem" in a["repair_rejected"] and "62.50" in a["repair_rejected"]


def test_a_repair_that_removes_the_failure_and_keeps_the_email_is_accepted(monkeypatch):
    agent = _agent(monkeypatch, good_model)
    _broken_writer(agent, monkeypatch)
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: b.replace("We will pay 61.25 GBP. ", ""))
    draft = _prompt(agent)
    assert draft["assurance"]["repaired"] is True and "61.25" not in draft["text"]


def test_a_reference_the_person_typed_in_their_request_is_theirs_to_use(monkeypatch):
    agent = _agent(monkeypatch, lambda s, u: "garbage")                      # classifier fails -> fallback family
    kinds = {v["kind"] for v in _prompt(agent)["assurance"]["violations"]}
    assert "ungrounded_reference" not in kinds                                # PO-77123 is in the request
