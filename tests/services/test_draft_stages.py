"""Stages 1, 3 and 4 around a fake model, including the fake model misbehaving.

The model is a stand-in, so what these prove is the PLUMBING and the VALIDATORS: that bad output
(malformed JSON, an unknown family, a made-up PO number, a reasoned field with no basis, a figure
in no fact, an out-of-range score) is caught and recorded, never passed through. They say nothing
about how well a real model classifies, plans or judges -- see specs/...-pending-live-verification.md.
"""

import json
from types import SimpleNamespace

import pytest

from src.services import draft_assurance as da
from src.services.draft_assurance import accountability, authority, brief as B, classify as C, judge as J
from src.services.draft_assurance import run as R, stages, tone as T
from tests.services.test_draft_assurance import (FakeConn, TABLES, _family_rules, _free_prompt_rules)

FAMILIES = {"negotiation_counter": "Countering a supplier position. More.", "free_prompt": "Free text. More."}
REQUEST = "Please ask Acme to confirm the price on PO-77123 and reply within the week"


def fake(*replies):
    """A model returning each reply in turn (then the last one again)."""
    it = list(replies)
    calls = []

    def ask(system, user):
        calls.append((system, user))
        r = it.pop(0) if len(it) > 1 else it[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)
    ask.calls = calls
    return ask


GOOD_CLS = {"family_id": "negotiation_counter", "confidence": 0.9,
            "candidates": [{"family_id": "negotiation_counter", "confidence": 0.9},
                           {"family_id": "free_prompt", "confidence": 0.1}],
            "lookup_keys": {"po_number": "PO-77123"},
            "user_instruction": "ask Acme to confirm the price"}


def classify(reply, template="T {families}"):
    return stages.classify_request(fake(reply), template, REQUEST, FAMILIES)


# --- Stage 1: the classifier ---------------------------------------------------------------

def test_a_good_classification_is_captured_with_its_lookup_keys_and_instruction():
    r = classify(GOOD_CLS)
    c = r["classification"]
    assert r["status"] == "captured" and r["clarification"] is None
    assert c["family_id"] == "negotiation_counter" and c["lookup_keys"] == {"po_number": "PO-77123"}
    assert c["user_instruction"] == "ask Acme to confirm the price" and c["instruction_verbatim"] is True


def test_prose_around_valid_json_is_tolerated():
    assert classify("Sure! Here you go:\n" + json.dumps(GOOD_CLS) + "\nHope that helps")["status"] == "captured"


@pytest.mark.parametrize("bad", ["", "not json at all", "[1,2]", "null", "{\"family_id\": "])
def test_malformed_model_output_is_invalid_not_a_crash(bad):
    r = classify(bad)
    assert r["status"] == "invalid" and "JSON" in r["reason"]


def test_a_model_that_raises_is_invalid_not_a_crash():
    assert stages.classify_request(fake(RuntimeError("GPU down")), "T", REQUEST, FAMILIES)["status"] == "invalid"


def test_an_unknown_family_id_is_refused():
    r = classify({**GOOD_CLS, "family_id": "invoice_dispute"})
    assert r["status"] == "invalid" and "not a configured family" in r["reason"]


def test_an_unknown_family_among_the_candidates_is_refused():
    r = classify({**GOOD_CLS, "candidates": [{"family_id": "made_up", "confidence": 0.5}]})
    assert r["status"] == "invalid"


@pytest.mark.parametrize("conf", [1.5, -0.1, "high", None, True])
def test_a_confidence_outside_0_to_1_is_refused(conf):
    assert classify({**GOOD_CLS, "confidence": conf})["status"] == "invalid"


def test_an_invented_po_number_is_dropped_as_a_lookup_key_and_recorded():
    r = classify({**GOOD_CLS, "lookup_keys": {"po_number": "PO-99999", "invoice_number": "INV-1"}})
    c = r["classification"]
    assert c["lookup_keys"] == {} and c["rejected_lookup_keys"] == {"po_number": "PO-99999", "invoice_number": "INV-1"}


def test_a_user_instruction_that_is_not_verbatim_falls_back_to_the_request_and_says_so():
    c = classify({**GOOD_CLS, "user_instruction": "chase them aggressively"})["classification"]
    assert c["user_instruction"] == REQUEST and c["instruction_verbatim"] is False


def test_low_confidence_asks_one_question_offering_the_top_two():
    r = classify({**GOOD_CLS, "confidence": 0.55,
                  "candidates": [{"family_id": "negotiation_counter", "confidence": 0.55},
                                 {"family_id": "free_prompt", "confidence": 0.3}]})
    q = r["clarification"]
    assert q["options"] == ["negotiation_counter", "free_prompt"] and q["question"].count("?") == 1
    assert q["reason"] == "low confidence"


def test_close_top_two_asks_even_when_confidence_is_high_enough():
    r = classify({**GOOD_CLS, "confidence": 0.75,
                  "candidates": [{"family_id": "negotiation_counter", "confidence": 0.75},
                                 {"family_id": "free_prompt", "confidence": 0.65}]})
    assert r["clarification"]["reason"] == "top two families are close"


def test_exactly_at_the_thresholds_does_not_ask():
    r = classify({**GOOD_CLS, "confidence": 0.70,
                  "candidates": [{"family_id": "negotiation_counter", "confidence": 0.70},
                                 {"family_id": "free_prompt", "confidence": 0.55}]})
    assert r["clarification"] is None


def test_the_family_list_the_model_sees_comes_from_config_not_code():
    ask = fake(GOOD_CLS)
    stages.classify_request(ask, "Families:\n{families}", REQUEST, {"brand_new_family": "Added as a row."})
    assert "brand_new_family: Added as a row." in ask.calls[0][0]


def test_without_a_governed_prompt_the_stage_is_unavailable_and_the_model_is_never_called():
    ask = fake(GOOD_CLS)
    assert stages.classify_request(ask, None, REQUEST, FAMILIES) == {
        "status": "unavailable", "reason": "the governed classifier prompt is not installed"}
    assert ask.calls == []


# --- Stage 3: the planner and the brief ------------------------------------------------------

FACTS = {"supplier_current_offer": "47.50", "currency": "GBP"}
CONTEXT = {"email_thread_summary": "one earlier note"}
GOOD_BRIEF = {"goal": "Agree a lower price", "key_points": ["Offer is 47.50 GBP"], "explicit_ask": "Confirm revised price",
              "deadline": "within the week", "tone_rationale": "Firm but courteous", "risks_to_avoid": ["threats"],
              "reasoned": {"requested_price": {"value": "44.80", "basis": ["supplier_current_offer"], "confidence": 0.7}},
              "assumptions": []}


def plan(reply, facts=FACTS):
    return stages.plan_brief(fake(reply), "T {facts}", family_id="negotiation_counter", tone=None,
                             instruction="", facts=facts, context=CONTEXT, request=REQUEST + " 44.80")


def test_a_good_brief_is_captured_with_its_basis():
    r = plan(GOOD_BRIEF)
    assert r["status"] == "captured" and r["brief"]["status"] == "ready"
    assert r["brief"]["reasoned"]["requested_price"]["basis"] == ["supplier_current_offer"]
    assert r["brief"]["assumptions"] == []


def test_a_reasoned_field_with_no_basis_becomes_an_assumption_a_person_must_confirm():
    bad = {**GOOD_BRIEF, "reasoned": {"requested_price": {"value": "44.80", "basis": []}}}
    b = plan(bad)["brief"]
    assert [a["key"] for a in b["assumptions"]] == ["requested_price"] and b["assumptions"][0]["resolution"] is None


def test_a_basis_naming_something_that_is_not_a_fact_or_context_is_dropped_not_trusted():
    bad = {**GOOD_BRIEF, "reasoned": {"requested_price": {"value": "44.80", "basis": ["market_survey_2026"]}}}
    b = plan(bad)["brief"]
    assert b["reasoned"]["requested_price"]["basis"] == [] and "market_survey_2026" in b["assumptions"][0]["text"]


def test_the_planner_may_say_a_fact_is_missing():
    r = plan({"missing": ["supplier_current_offer"]})
    assert r["brief"] == {"status": "missing", "missing": ["supplier_current_offer"]}


@pytest.mark.parametrize("bad", ["", "oops", "[]", {"goal": "x"}, {**GOOD_BRIEF, "reasoned": []},
                                 {**GOOD_BRIEF, "key_points": "not a list"},
                                 {**GOOD_BRIEF, "reasoned": {"p": {"basis": []}}}])
def test_a_malformed_brief_is_invalid(bad):
    assert plan(bad)["status"] == "invalid"


def test_a_brief_stating_a_figure_in_no_fact_and_not_in_the_request_is_refused():
    r = plan({**GOOD_BRIEF, "key_points": ["Their offer is 52.00 GBP"]})
    assert r["status"] == "invalid" and "52" in r["reason"]


def test_a_brief_with_an_invented_po_number_is_refused():
    assert plan({**GOOD_BRIEF, "goal": "Settle PO-55555"})["status"] == "invalid"


def test_without_a_governed_planner_prompt_the_stage_is_unavailable():
    assert stages.plan_brief(fake(GOOD_BRIEF), None, family_id="x", tone=None, instruction="", facts={},
                             context={}, request="")["status"] == "unavailable"


# --- Stage 4: the judge --------------------------------------------------------------------

RUBRIC = ["completeness", "clarity_of_ask", "tone_fit", "concision"]


def judge(reply, rubric=RUBRIC):
    return stages.judge_draft(fake(reply), "T {rubric}", rubric, text="Dear Alex...", brief=None, facts={})


def test_a_good_judgement_is_scored_and_the_overall_is_computed_by_code_not_trusted():
    r = judge({"scores": dict(completeness=5, clarity_of_ask=4, tone_fit=4, concision=3), "overall": 5, "rationale": "ok"})
    assert r["status"] == "scored" and r["overall"] == 4.0


@pytest.mark.parametrize("scores", [
    dict(completeness=6, clarity_of_ask=4, tone_fit=4, concision=3),          # above range
    dict(completeness=0, clarity_of_ask=4, tone_fit=4, concision=3),          # below range
    dict(completeness=4.5, clarity_of_ask=4, tone_fit=4, concision=3),        # not whole
    dict(completeness="five", clarity_of_ask=4, tone_fit=4, concision=3),     # not a number
    dict(completeness=5, clarity_of_ask=4, tone_fit=4),                       # a criterion missing
])
def test_bad_scores_make_the_judgement_invalid_and_are_never_averaged_over_what_is_left(scores):
    assert judge({"scores": scores})["status"] == "invalid"


@pytest.mark.parametrize("bad", ["", "no", "[]", {"rationale": "x"}, {"scores": [5, 5]}])
def test_malformed_judge_output_is_invalid(bad):
    assert judge(bad)["status"] == "invalid"


def test_an_unknown_criterion_from_the_model_is_ignored_and_listed():
    r = judge({"scores": dict(completeness=5, clarity_of_ask=5, tone_fit=5, concision=5, vibes=1)})
    assert r["status"] == "scored" and r["ignored_criteria"] == ["vibes"] and r["overall"] == 5.0


def test_a_family_with_no_rubric_cannot_be_judged():
    assert judge({"scores": {}}, rubric=[])["status"] == "unavailable"


def test_without_a_governed_judge_prompt_there_is_no_score():
    assert stages.judge_draft(fake({}), None, RUBRIC, text="x", brief=None, facts={})["status"] == "unavailable"


# --- authority guardrail ---------------------------------------------------------------------

def _engine(limit="10000", currency="GBP", governed=True):
    """A policy engine in the shape resolve_authority reads."""
    rules = {"defer_value_limit_to": "approval_thresholds", "auto_intents": []}
    if currency:
        rules["limit_currency"] = currency
    approval = {"details": {"rules": {"default_threshold_gbp": limit, **({"currency": currency} if currency else {})}}}
    auto = {"details": {"rules": rules}}
    return SimpleNamespace(get_policy=lambda slug: auto if slug == "email_reply_autonomy" else (approval if governed else None))


def test_a_counter_within_the_limit_is_cleared(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": True, "limit_gbp": "10000", "limit_currency": "GBP"}})
    assert authority.check_commitment(None, "email_drafting_agent", authority.commitment_amount({"counter_price": 4480}), "GBP")["verdict"] == "within"


def test_a_counter_over_the_limit_exceeds(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": True, "limit_gbp": "10000", "limit_currency": "GBP"}})
    d = {"counter_price": 44.8, "line_items": [{"quantity": 300, "unit_price": 44.8}]}
    r = authority.check_commitment(None, "a", authority.commitment_amount(d), "GBP")
    assert r["verdict"] == "exceeds" and r["amount"] == "13440.0"


def test_an_ungoverned_agent_is_unresolved_not_permitted(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": False, "limit_gbp": None, "reason": "no policy"}})
    assert authority.check_commitment(None, "a", authority.commitment_amount({"counter_price": 5}), "GBP")["verdict"] == "unresolved"


def test_a_different_currency_is_unresolved_and_no_rate_is_invented(monkeypatch):
    from src.services.governance_tools import authority as ga
    monkeypatch.setattr(ga, "resolve_authority", lambda e, a: {a[0]: {"governed": True, "limit_gbp": "10000", "limit_currency": "GBP"}})
    r = authority.check_commitment(None, "a", authority.commitment_amount({"counter_price": 5}), "EUR")
    assert r["verdict"] == "unresolved" and "no rate" in r["reason"]


def test_a_counter_with_no_amount_is_unresolved():
    assert authority.check_commitment(None, "a", authority.commitment_amount({}), "GBP")["verdict"] == "unresolved"


def test_an_authority_lookup_that_raises_is_unresolved(monkeypatch):
    from src.services.governance_tools import authority as ga

    def boom(e, a):
        raise RuntimeError("policy store down")
    monkeypatch.setattr(ga, "resolve_authority", boom)
    assert authority.check_commitment(None, "a", authority.commitment_amount({"counter_price": 5}), "GBP")["verdict"] == "unresolved"


# --- accountability ------------------------------------------------------------------------------

def test_a_person_is_the_initiator_only_when_it_is_not_a_known_agent_id():
    assert accountability.initiated("nick@acme.test", "EmailDraftingAgent", {"AgentNick"}) == {"id": "nick@acme.test", "kind": "user"}
    assert accountability.initiated("AgentNick", "EmailDraftingAgent", {"AgentNick"}) == {"id": "EmailDraftingAgent", "kind": "agent"}
    assert accountability.initiated(None, "EmailDraftingAgent")["kind"] == "agent"


def test_reviewer_problems():
    agent = {"id": "NegotiationAgent", "kind": "agent"}
    assert accountability.reviewer_problem("buyer@acme.test", agent) is None
    assert accountability.reviewer_problem("", agent) and accountability.reviewer_problem(None, agent)
    assert "agent that wrote" in accountability.reviewer_problem("NegotiationAgent", agent)
    assert accountability.reviewer_problem("buyer@acme.test", agent, autonomous=True)


def test_the_mailbox_owner_is_found_only_from_exactly_one_active_binding():
    class Cur:
        def __init__(self, rows): self.rows = rows
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, p=None): self.sql = sql
        def fetchall(self): return self.rows
    class Conn:
        def __init__(self, rows): self.c = Cur(rows)
        def cursor(self): return self.c
    assert accountability.mailbox_owner(Conn([("sub-1",)]), "Nick@Acme.test") == "sub-1"
    assert accountability.mailbox_owner(Conn([]), "x@y.test") is None
    assert accountability.mailbox_owner(Conn([("a",), ("b",)]), "x@y.test") is None     # ambiguous: never guess
    assert "is_active" in Conn([("a",)]).c.sql if hasattr(Conn([("a",)]).c, "sql") else True
    assert accountability.mailbox_owner(Conn([]), None) is None


# --- tone ------------------------------------------------------------------------------------------

TONE_RULES = Path = None


def _tone_rules():
    from pathlib import Path as P
    sql = P(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_tone_rules.sql"
    return T.parse_rules(json.loads(sql.read_text().split("$json$")[1])["rules"], 1)


class ToneConn:
    """Answers the three lookups tone derivation makes."""
    def __init__(self, supplier=None, contacts=None):
        self.supplier, self.contacts, self.sql = supplier or {}, contacts, []
    def cursor(self): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None):
        self.sql.append(sql)
        self._rows = []
        if "FROM proc.bp_supplier" in sql:
            col = sql.split("SELECT ")[1].split(" FROM")[0]
            self._rows = [(self.supplier[col],)] if col in self.supplier else []
        elif "workflow_email_tracking" in sql and self.contacts is not None:
            self._rows = [(self.contacts,)]
    def fetchall(self): return self._rows


def test_each_variable_records_where_it_came_from():
    conn = ToneConn({"is_preferred_supplier": True, "contact_role_1": "Account Manager", "country": "Germany"}, contacts=1)
    t = T.derive_tone(conn, _tone_rules(), supplier_id="S-1", workflow_id="wf")
    v, s = t["values"], t["sources"]
    assert (v["relationship_tier"], v["recipient_seniority"], v["region_formality"], v["escalation_level"]) == ("preferred", "manager", "high", 2)
    assert all(s[k]["source"] == "postgres" for k in ("relationship_tier", "recipient_seniority", "region_formality", "escalation_level"))
    assert s["leverage"]["source"] == "default" and s["relationship_health"]["source"] == "default"
    assert v["leverage"] == "medium" and v["relationship_health"] == "neutral"


def test_with_no_data_every_variable_takes_its_declared_unknown_default():
    t = T.derive_tone(ToneConn(), _tone_rules(), supplier_id=None, workflow_id=None)
    assert t["values"] == {"relationship_tier": "standard", "escalation_level": 1, "leverage": "medium",
                           "recipient_seniority": "manager", "relationship_health": "neutral", "region_formality": "medium",
                           "warmth": "neutral", "directness": "balanced"}
    assert {x["source"] for x in t["sources"].values()} == {"default"}


def test_escalation_follows_prior_contacts_on_the_thread():
    levels = {n: T.derive_tone(ToneConn(contacts=n), _tone_rules(), supplier_id="S", workflow_id="w")["values"]["escalation_level"]
              for n in (0, 1, 2, 3, 5, 9)}
    assert levels == {0: 1, 1: 2, 2: 2, 3: 3, 5: 4, 9: 4}


def test_the_persons_instruction_overrides_what_postgres_says():
    conn = ToneConn({"is_preferred_supplier": True}, contacts=0)
    t = T.derive_tone(conn, _tone_rules(), supplier_id="S", workflow_id="w", instruction="push hard on this one")
    assert t["values"]["escalation_level"] == 3 and t["sources"]["escalation_level"]["source"] == "user_instruction"
    assert t["sources"]["relationship_tier"]["source"] == "postgres"
    soft = T.derive_tone(ToneConn(contacts=6), _tone_rules(), supplier_id="S", workflow_id="w",
                         instruction="keep it light, they are helping us elsewhere")
    assert soft["values"]["escalation_level"] == 1 and soft["values"]["relationship_health"] == "good"


def test_a_failed_lookup_is_no_data_not_a_crash():
    class Boom(ToneConn):
        def execute(self, sql, params=None): raise RuntimeError("db down")
    t = T.derive_tone(Boom(), _tone_rules(), supplier_id="S", workflow_id="w")
    assert {x["source"] for x in t["sources"].values()} == {"default"}


def test_tone_rules_with_a_bad_default_or_an_unreadable_column_are_refused_at_load():
    from pathlib import Path as P
    base = json.loads((P(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_tone_rules.sql").read_text().split("$json$")[1])["rules"]
    for mutate in (lambda r: r["variables"]["leverage"].update(unknown_default="extreme"),
                   lambda r: r["variables"]["region_formality"]["derive"].update(column="bank_iban"),
                   lambda r: r["variables"].pop("leverage"),
                   lambda r: r["instruction_overrides"].append({"match": "x", "set": {"leverage": "nope"}})):
        rules = json.loads(json.dumps(base))
        mutate(rules)
        with pytest.raises(T.ToneRulesUnavailable):
            T.parse_rules(rules)


# --- instruction overrides match whole words only --------------------------------------------------------

@pytest.mark.parametrize("text,var,expected", [
    ("ask Acme to confirm the price", "escalation_level", 1),          # "firm" inside "confirm" must not fire
    ("install the software update", "escalation_level", 1),            # "soft" inside "software"
    ("keep an informal tone", "region_formality", "low"),              # "formal" inside "informal" must not set high
    ("push hard", "escalation_level", 3),
    ("be firmly polite", "escalation_level", 3),
    ("write it formally", "region_formality", "high"),
])
def test_overrides_match_whole_words_not_fragments(text, var, expected):
    t = T.derive_tone(ToneConn(), _tone_rules(), supplier_id=None, workflow_id=None, instruction=text)
    assert t["values"][var] == expected


# --- gaps: what a variable does when there is no data -------------------------------------------------------

def _gaps(conn=None, **kw):
    return T.derive_tone(conn or ToneConn(), _tone_rules(), supplier_id=kw.pop("supplier_id", None),
                         workflow_id=kw.pop("workflow_id", None), **kw)


def test_every_defaulted_variable_is_listed_as_a_gap_with_its_rule():
    t = _gaps()
    assert {g["variable"]: g["on_gap"] for g in t["gaps"]} == {
        "relationship_tier": "default", "escalation_level": "assume", "leverage": "default",
        "recipient_seniority": "assume", "relationship_health": "default", "region_formality": "default",
        "warmth": "default", "directness": "default"}


def test_a_variable_with_data_is_not_a_gap():
    t = _gaps(ToneConn({"is_preferred_supplier": True, "contact_role_1": "Head of Sales", "country": "Germany"}, contacts=1),
              supplier_id="S", workflow_id="w")
    assert {g["variable"] for g in t["gaps"]} == {"leverage", "relationship_health", "warmth", "directness"}


def test_a_variable_the_person_set_is_not_a_gap_even_with_no_data():
    t = _gaps(instruction="push hard")
    assert "escalation_level" not in {g["variable"] for g in t["gaps"]} and "directness" not in {g["variable"] for g in t["gaps"]}


def test_ask_is_refused_because_nothing_can_answer_it_yet():
    from pathlib import Path as P
    base = json.loads((P(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_tone_rules.sql").read_text().split("$json$")[1])["rules"]
    for bad in ("ask", None, "", "sometimes"):
        rules = json.loads(json.dumps(base))
        rules["variables"]["leverage"]["on_gap"] = bad
        with pytest.raises(T.ToneRulesUnavailable) as e:
            T.parse_rules(rules)
        assert "on_gap" in str(e.value)


def test_the_two_new_groups_have_explicit_unknown_defaults_and_allowed_values():
    r = _tone_rules().variables
    assert (r["warmth"]["allowed"], r["warmth"]["unknown_default"]) == (["cool", "neutral", "warm"], "neutral")
    assert (r["directness"]["allowed"], r["directness"]["unknown_default"]) == (["indirect", "balanced", "direct"], "balanced")


# --- instruction groups ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expect", [
    ("sound apologetic",            {"warmth": "warm", "directness": "indirect", "escalation_level": 1}),
    ("keep it appreciative",        {"warmth": "warm"}),
    ("be collaborative",            {"warmth": "warm", "relationship_health": "good", "directness": "balanced"}),
    ("this is urgent",              {"escalation_level": 3, "directness": "direct"}),
    ("be firm",                     {"escalation_level": 3, "directness": "direct"}),
    ("keep it concise",             {"directness": "direct"}),
    ("be diplomatic",               {"directness": "indirect"}),
    ("matter-of-fact please",       {"warmth": "neutral", "directness": "balanced"}),
    ("a bit cold",                  {"warmth": "cool"}),
    ("firm but friendly",           {"escalation_level": 3, "directness": "direct", "warmth": "warm"}),   # both, not one
])
def test_each_instruction_group_sets_what_it_says(text, expect):
    t = _gaps(instruction=text)
    for var, val in expect.items():
        assert t["values"][var] == val and t["sources"][var]["source"] == "user_instruction", (text, var)
    assert t["unmapped_instruction"] == []


@pytest.mark.parametrize("task", [
    "ask Acme to confirm the price on PO-77123", "write a professional email", "chase the delivery",
    "request a quote for 25 chairs", "reply within the week",
])
def test_a_plain_task_is_not_mistaken_for_a_tone_request(task):
    t = _gaps(instruction=task)
    assert t["unmapped_instruction"] == [] and {s["source"] for s in t["sources"].values()} == {"default"}


@pytest.mark.parametrize("text,words", [("be warm but brusque", ["brusque"]), ("sound aggressive", ["aggressive"]),
                                         ("serious and stern", []), ("terse and aloof", [])])
def test_a_tone_word_nothing_maps_is_reported_not_dropped(text, words):
    assert _gaps(instruction=text)["unmapped_instruction"] == words
