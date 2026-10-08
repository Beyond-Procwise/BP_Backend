"""A whole assurance run: family resolved, tone derived, stages recorded with explicit empties."""

import json
from types import SimpleNamespace

import pytest

from src.services.draft_assurance import FamilyConfigUnavailable, run as R
from tests.services.test_draft_assurance import FakeConn, TABLES, _family_rules, _free_prompt_rules
from tests.services.test_draft_stages import FAMILIES, GOOD_BRIEF, GOOD_CLS, fake, _tone_rules
from pathlib import Path

GOOD = ("Thank you for your offer of 47.50 GBP. We propose 44.80 GBP. "
        "Please confirm by 30 October 2026?")
TONE_SQL = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_tone_rules.sql"
FAMILY_V2 = {"rubric": ["ask_is_specific", "deadline_stated"], "authority_agent": "email_drafting_agent"}


class Conn(FakeConn):
    """Fact tables, plus the supplier and contact lookups tone derivation makes."""
    supplier = {"is_preferred_supplier": True, "contact_role_1": "Director of Sales", "country": "Germany"}
    contacts = 2

    def cursor(self):
        cur = super().cursor()
        orig = cur.execute

        def execute(sql, params=None):
            if "FROM proc.bp_supplier WHERE supplier_id" in sql and "SELECT contact_name_1" not in sql and ", " not in sql.split("FROM")[0]:
                col = sql.split("SELECT ")[1].split(" FROM")[0]
                cur.conn.log.append(sql)
                cur.rows = [(self.supplier[col],)] if col in self.supplier else []
            elif "workflow_email_tracking" in sql:
                cur.conn.log.append(sql)
                cur.rows = [(self.contacts,)]
            else:
                orig(sql, params)
        cur.execute = execute
        return cur


def _policies(with_tone=True, counter=True, free=True):
    rows = {}
    if counter:
        rows["email_family_negotiation_counter"] = {"details": {"rules": {**_family_rules(), **FAMILY_V2}}, "version": 1, "policy_desc": FAMILIES["negotiation_counter"]}
    if free:
        rows["email_family_free_prompt"] = {"details": {"rules": {**_free_prompt_rules(), "rubric": ["completeness"]}}, "version": 1, "policy_desc": FAMILIES["free_prompt"]}
    if with_tone:
        rows["email_tone_rules"] = {"details": {"rules": json.loads(TONE_SQL.read_text().split("$json$")[1])["rules"]}, "version": 1}
    listing = [{"policy_type": "email_family", "details": r["details"], "policy_desc": r["policy_desc"]}
               for k, r in rows.items() if k.startswith("email_family_")]
    return SimpleNamespace(get_policy=lambda slug: rows.get(slug), list_policies=lambda: listing)


def env(ask=None, prompts=None, user=None, exemplars=lambda: {"ids": [], "scope": "none"}, **kw):
    prompts = {"email_family_classify": "C {families}", "email_brief_plan": "P {facts}", "email_draft_judge": "J {rubric}"} \
        if prompts is None else prompts
    return R.Env(conn_factory=lambda: Conn(kw.pop("tables", TABLES)), policy_engine=kw.pop("engine", _policies()), ask=ask,
                 prompt=lambda n: prompts.get(n), master_emails=lambda sid: ["a@x.test"],
                 agent_name="NegotiationAgent", agent_ids={"AgentNick"}, user_id=user, exemplars=exemplars)


DATA = {"supplier_id": "S-1", "workflow_id": "wf-1", "current_offer": 47.5, "currency": "GBP", "counter_price": 44.8,
        "response_deadline": "30 October 2026", "asks": ["Confirm revised price"], "rationale": "Anchor below their offer",
        "reasoned_basis": {"response_deadline": ["email_thread_summary"]}}
JUDGE_OK = {"scores": {"ask_is_specific": 5, "deadline_stated": 4}, "rationale": "clear"}


def _declared(ask=None, **kw):
    r = R.begin(env(ask or fake(JUDGE_OK), **kw), dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1")
    return r, r.finalize(GOOD, ["a@x.test"], "S-1")


# --- declared families --------------------------------------------------------------------------

def test_a_declared_family_is_recorded_as_declared_and_not_classified():
    _, rec = _declared()
    assert rec["family_source"] == "declared" and rec["classification"] is None and rec["clarification"] is None
    assert rec["stage_status"]["classify"] == {"status": "not_run", "reason": "family declared by the calling path"}


def test_a_declared_family_that_is_not_in_config_is_refused_and_the_draft_is_unassured():
    run = R.begin(env(), dict(DATA), slug="email_family_no_such_family", workflow_id="wf-1")
    rec = run.finalize(GOOD, [], "S-1")
    assert rec["status"] == "unassured" and "unknown or unreadable family" in rec["reason"]
    assert rec["family_source"] == "declared" and rec["ready"] is False


def test_a_declared_draft_still_gets_facts_validation_and_a_judgement():
    _, rec = _declared()
    assert rec["facts"]["supplier_current_offer"]["row_id"] == "2" and rec["violations"] == []
    assert rec["judge"]["status"] == "scored" and rec["judge"]["overall"] == 4.5


# --- Stage 1 outputs ----------------------------------------------------------------------------------

def test_tone_variables_and_their_sources_are_recorded():
    _, rec = _declared()
    assert rec["tone"]["variables"]["region_formality"] == "high"
    assert rec["tone"]["variables"]["escalation_level"] == 2 and rec["tone"]["sources"]["escalation_level"]["source"] == "postgres"
    assert rec["tone"]["sources"]["leverage"]["source"] == "default"
    assert rec["stage_status"]["tone"]["status"] == "captured"


def test_the_user_instruction_overrides_tone_and_is_recorded_verbatim():
    run = R.begin(env(fake(JUDGE_OK)), dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1",
                  instruction="keep it light, they are helping us elsewhere")
    rec = run.finalize(GOOD, ["a@x.test"], "S-1")
    assert rec["tone"]["variables"]["escalation_level"] == 1
    assert rec["tone"]["sources"]["escalation_level"]["source"] == "user_instruction"
    assert rec["user_instruction"] == "keep it light, they are helping us elsewhere"


def test_without_tone_rules_the_tone_is_unavailable_and_says_why_not_silently_empty():
    _, rec = _declared(engine=_policies(with_tone=False))
    assert rec["tone"] is None and rec["stage_status"]["tone"]["status"] == "unavailable"
    assert "no tone rules" in rec["stage_status"]["tone"]["reason"]


# --- Stage 2 outputs: explicit empty vs not captured -----------------------------------------------------

def test_exemplar_retrieval_that_found_nothing_stores_an_explicit_empty_list():
    _, rec = _declared(exemplars=lambda: {"ids": [], "scope": "none"})
    assert rec["exemplars"] == {"ids": [], "scope": "none"} and rec["stage_status"]["exemplars"]["status"] == "empty"


def test_exemplars_that_were_used_are_listed_with_their_scope():
    _, rec = _declared(exemplars=lambda: {"ids": [4, 9], "scope": "organisation"})
    assert rec["exemplars"] == {"ids": [4, 9], "scope": "organisation"} and rec["stage_status"]["exemplars"]["status"] == "captured"


def test_retrieval_that_never_ran_is_not_captured_rather_than_empty():
    _, rec = _declared(exemplars=lambda: None)
    assert rec["exemplars"] is None and rec["stage_status"]["exemplars"]["status"] == "not_run"


def test_retrieval_that_failed_is_unavailable_and_never_pretends_to_be_empty():
    def boom():
        raise RuntimeError("store down")
    _, rec = _declared(exemplars=boom)
    assert rec["stage_status"]["exemplars"]["status"] == "unavailable"


# --- Stage 3: the counter brief is mapped, not invented -----------------------------------------------------

def test_the_counter_brief_maps_the_negotiation_agents_own_output():
    _, rec = _declared()
    b = rec["brief"]
    assert b["goal"] == "Anchor below their offer" and b["explicit_ask"] == "Confirm revised price"
    assert b["reasoned"]["counter_price"]["basis"] == ["supplier_current_offer"]
    assert b["deadline"] == "30 October 2026" and "walkaway_price" in b["risks_to_avoid"]
    assert "escalation_level 2 (postgres)" in b["tone_rationale"]


def test_where_the_negotiation_output_gives_no_basis_the_brief_says_assumption_or_missing():
    data = {k: v for k, v in DATA.items() if k not in ("rationale", "strategy", "asks", "reasoned_basis")}
    run = R.begin(env(fake(JUDGE_OK)), data, slug="email_family_negotiation_counter", workflow_id="wf-1")
    rec = run.finalize(GOOD, ["a@x.test"], "S-1")
    b = rec["brief"]
    assert set(b["missing"]) >= {"goal", "explicit_ask"} and b["goal"] is None and b["explicit_ask"] is None
    assert [a["key"] for a in rec["assumption_items"]] == ["response_deadline"]
    assert rec["ready"] is False                                   # an unresolved assumption blocks readiness


def test_a_draft_with_every_assumption_grounded_is_ready():
    _, rec = _declared()
    assert rec["assumption_items"] == [] and rec["ready"] is True


# --- Stage 4 and authority ----------------------------------------------------------------------------------------

def test_a_judge_that_returns_garbage_is_recorded_invalid_not_scored():
    _, rec = _declared(ask=fake("lol no"))
    assert rec["judge"]["status"] == "invalid" and rec["stage_status"]["judge"]["status"] == "invalid"


def test_with_no_model_the_judge_is_unavailable_and_no_score_is_invented():
    _, rec = _declared(ask=None)
    run = R.begin(env(None), dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1")
    rec = run.finalize(GOOD, ["a@x.test"], "S-1")
    assert rec["judge"] == {"status": "unavailable", "reason": "no model is available"}


def test_the_authority_guardrail_runs_on_every_counter_and_is_recorded(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": True, "limit_gbp": "100", "limit_currency": "GBP"}})
    _, rec = _declared()
    assert rec["authority"]["verdict"] == "within" and rec["stage_status"]["authority"]["status"] == "captured"
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": True, "limit_gbp": "10", "limit_currency": "GBP"}})
    _, rec = _declared()
    assert rec["authority"]["verdict"] == "exceeds"


def test_an_unresolvable_authority_is_recorded_not_skipped(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": False, "limit_gbp": None, "reason": "no limit"}})
    _, rec = _declared()
    assert rec["authority"]["verdict"] == "unresolved"


# --- accountability ---------------------------------------------------------------------------------------------------

def test_the_initiator_is_the_agent_unless_a_person_asked():
    _, rec = _declared()
    assert rec["accountability"] == {"initiated_by": "NegotiationAgent", "kind": "agent"}
    run = R.begin(env(fake(JUDGE_OK), user="nick@acme.test"), dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1")
    assert run.finalize(GOOD, [], "S-1")["accountability"] == {"initiated_by": "nick@acme.test", "kind": "user"}
    run = R.begin(env(fake(JUDGE_OK), user="AgentNick"), dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1")
    assert run.finalize(GOOD, [], "S-1")["accountability"]["kind"] == "agent"


# --- free text: classified, fallback, clarification ----------------------------------------------------------------

REQ = "Please ask Acme to confirm the price on PO-77123 and reply within the week"
PLAN = {**GOOD_BRIEF, "reasoned": {"requested_price": {"value": "44.80", "basis": ["supplier_current_offer"], "confidence": 0.7}},
        "key_points": ["Their offer is 47.5"]}
CLS = {**GOOD_CLS, "user_instruction": "ask Acme to confirm the price"}


def _prompt_run(*replies, **kw):
    data = {"prompt": REQ, "supplier_id": "S-1", "workflow_id": "wf-1"}
    r = R.begin(env(fake(*replies), **kw), data, slug=None, workflow_id="wf-1", request=REQ, classify=True)
    return r, r.finalize("Please confirm the price within the week?", ["a@x.test"], "S-1")


def test_free_text_is_classified_planned_and_judged():
    r, rec = _prompt_run(CLS, PLAN, {"scores": {"ask_is_specific": 4, "deadline_stated": 4}})
    assert rec["family_source"] == "classified" and rec["family_id"] == "negotiation_counter"
    assert rec["classification"]["lookup_keys"] == {"po_number": "PO-77123"}
    assert rec["user_instruction"] == "ask Acme to confirm the price"
    assert rec["clarification"] == {} and rec["stage_status"]["classify"]["status"] == "captured"
    assert rec["brief"]["status"] == "ready" and rec["stage_status"]["brief"]["status"] == "captured"
    assert r.planned["status"] == "captured"


def test_a_malformed_classification_falls_back_to_the_conservative_family_and_says_so():
    r, rec = _prompt_run("garbage", PLAN)
    assert rec["family_source"] == "fallback" and rec["family_id"] == "free_prompt"
    assert rec["stage_status"]["classify"]["status"] == "invalid" and rec["classification"] is None


def test_an_unknown_family_from_the_model_falls_back_and_is_never_used():
    r, rec = _prompt_run({**CLS, "family_id": "wire_transfer_request"}, PLAN)
    assert rec["family_id"] == "free_prompt" and rec["family_source"] == "fallback"
    assert "not a configured family" in rec["stage_status"]["classify"]["reason"]


def test_an_unavailable_classifier_falls_back():
    r, rec = _prompt_run(CLS, PLAN, prompts={})
    assert rec["family_source"] == "fallback" and rec["stage_status"]["classify"]["status"] == "unavailable"


def test_low_confidence_leaves_a_clarification_open_and_the_draft_not_ready():
    low = {**CLS, "confidence": 0.5, "candidates": [{"family_id": "negotiation_counter", "confidence": 0.5},
                                                      {"family_id": "free_prompt", "confidence": 0.4}]}
    r, rec = _prompt_run(low, PLAN)
    assert rec["clarification"]["options"] == ["negotiation_counter", "free_prompt"] and rec["ready"] is False


def _contact(rec):
    return (rec["facts"].get("supplier_contact_name") or {}).get("row_id")


def test_a_candidate_lookup_key_never_overrides_what_the_caller_supplied():
    req = REQ + " (supplier S-2)"
    cls = {**CLS, "lookup_keys": {"supplier_id": "S-2"}}
    data = {"prompt": req, "supplier_id": "S-1", "workflow_id": "wf-1"}
    r = R.begin(env(fake(cls, PLAN)), data, slug=None, workflow_id="wf-1", request=req, classify=True)
    rec = r.finalize("Please confirm.", ["a@x.test"], "S-1")
    assert rec["classification"]["lookup_keys"] == {"supplier_id": "S-2"}     # the model did propose S-2 ...
    assert _contact(rec) == "S-1"                                              # ... and the caller's S-1 was used


def test_a_candidate_lookup_key_fills_a_gap_the_caller_left():
    req = REQ + " (supplier S-1)"
    cls = {**CLS, "lookup_keys": {"supplier_id": "S-1"}}
    data = {"prompt": req, "workflow_id": "wf-1"}                             # no supplier_id supplied
    r = R.begin(env(fake(cls, PLAN)), data, slug=None, workflow_id="wf-1", request=req, classify=True)
    assert _contact(r.finalize("Please confirm.", ["a@x.test"], "S-1")) == "S-1"


def test_an_invented_po_number_from_the_classifier_is_not_used_as_a_key():
    r, rec = _prompt_run({**CLS, "lookup_keys": {"po_number": "PO-00000"}}, PLAN)
    assert rec["classification"]["lookup_keys"] == {} and rec["classification"]["rejected_lookup_keys"] == {"po_number": "PO-00000"}


def test_a_brief_with_a_figure_in_no_fact_is_recorded_invalid_and_not_used():
    r, rec = _prompt_run(CLS, {**PLAN, "key_points": ["Their offer is 99.00"]})
    assert rec["brief"] is None and rec["stage_status"]["brief"]["status"] == "invalid"


def test_every_stage_that_did_not_produce_a_value_says_so_in_stage_status():
    r, rec = _prompt_run("garbage", "garbage", "garbage", prompts={}, exemplars=lambda: None)
    st = rec["stage_status"]
    assert {k: v["status"] for k, v in st.items()} == {
        "classify": "unavailable", "tone": "captured", "exemplars": "not_run",
        "brief": "unavailable", "judge": "unavailable", "authority": "not_run", "steering": "not_run"}


# --- tone gaps reach the reviewer ----------------------------------------------------------------------------------------

class SparseConn(Conn):
    """A supplier we know nothing about, on a thread with no history."""
    supplier = {}
    contacts = None

    def cursor(self):
        cur = super().cursor()
        orig = cur.execute

        def execute(sql, params=None):
            if "workflow_email_tracking" in sql:
                cur.conn.log.append(sql); cur.rows = []
            else:
                orig(sql, params)
        cur.execute = execute
        return cur


def _sparse(instruction=None, **kw):
    e = env(fake(JUDGE_OK))
    e.conn_factory = lambda: SparseConn(TABLES)
    run = R.begin(e, dict(DATA), slug="email_family_negotiation_counter", workflow_id="wf-1", instruction=instruction)
    return run.finalize(GOOD, ["a@x.test"], "S-1")


def test_an_assume_gap_becomes_something_the_reviewer_must_confirm_and_blocks_ready():
    rec = _sparse()
    ids = {a["id"] for a in rec["assumption_items"]}
    assert {"tone:escalation_level", "tone:recipient_seniority"} <= ids
    assert rec["ready"] is False
    assert "assumed to be 1" in next(a["text"] for a in rec["assumption_items"] if a["id"] == "tone:escalation_level")


def test_a_default_gap_is_recorded_but_does_not_ask_anything():
    rec = _sparse()
    ids = {a["id"] for a in rec["assumption_items"]}
    assert not {"tone:leverage", "tone:warmth", "tone:directness", "tone:region_formality", "tone:relationship_health"} & ids
    assert rec["tone"]["sources"]["leverage"]["source"] == "default"


def test_a_gap_the_instruction_filled_is_not_asked_about():
    rec = _sparse("push hard")
    assert "tone:escalation_level" not in {a["id"] for a in rec["assumption_items"]}
    assert "tone:recipient_seniority" in {a["id"] for a in rec["assumption_items"]}


def test_confirming_the_defaults_makes_the_draft_ready():
    from tests.services.test_draft_events import MemDb
    rec = _sparse()
    from src.services.draft_assurance import capture
    db = MemDb(); db.add_capture(assumption_items=rec["assumption_items"], ready=rec["ready"])
    done = capture.confirm_assumptions(db, "U-1", [{"id": a["id"], "action": "confirm"} for a in rec["assumption_items"]], "nick")
    assert done["ready"] is True


def test_an_instruction_nothing_understood_is_shown_not_ignored():
    rec = _sparse("be warm but brusque")
    item = next(a for a in rec["assumption_items"] if a["id"] == "tone:instruction")
    assert "'brusque'" in item["text"] and item["resolution"] is None and rec["ready"] is False
    assert rec["tone"]["variables"]["warmth"] == "warm"          # the part it did understand still applied


def test_a_plain_task_instruction_raises_no_tone_question():
    assert "tone:instruction" not in {a["id"] for a in _sparse("ask Acme to confirm the price")["assumption_items"]}


def test_the_classifier_question_appears_as_an_item_the_reviewer_can_answer():
    low = {**CLS, "confidence": 0.5, "candidates": [{"family_id": "negotiation_counter", "confidence": 0.5},
                                                      {"family_id": "free_prompt", "confidence": 0.4}]}
    r, rec = _prompt_run(low, PLAN)
    item = next(a for a in rec["assumption_items"] if a["id"] == "clarification")
    assert item["options"] == ["negotiation_counter", "free_prompt"] and "?" in item["text"] and rec["ready"] is False


def test_with_no_tone_rules_there_are_no_tone_questions_and_the_stage_says_why():
    rec = _declared(engine=_policies(with_tone=False))[1]
    assert not [a for a in rec["assumption_items"] if a["id"].startswith("tone:")]
    assert rec["stage_status"]["tone"]["status"] == "unavailable"
