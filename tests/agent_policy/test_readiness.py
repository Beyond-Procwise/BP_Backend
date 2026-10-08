import copy

import pytest

from services.agent_policy import readiness as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

DOC_TEXT = "1.1 Refunds or credits above $500 need approval from the Finance Manager.\n1.2 ..."


MAPPED = {n: {"groups": ["g"], "emails": []} for n in ("Finance Manager", "CFO", "Procurement Lead")}


@pytest.fixture(autouse=True)
def _decider_map(monkeypatch):
    """Existing tests are about other fields: give them a fully linked decider map."""
    monkeypatch.setattr(R, "_load_deciders", lambda: MAPPED)


def _ready_form():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["hidden"]["condition"]["all"][0]["value"] = ["refund.issue", "credit.issue"]
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    return form


def _fields(problems):
    return [p["field"] for p in problems]


def test_complete_policy_has_no_problems():
    assert R.activation_problems(_ready_form(), REGISTRY, SETTINGS) == []


def test_draft_with_only_a_name_lists_every_problem_in_form_order():
    problems = R.activation_problems({"name": "Only a name"}, REGISTRY, SETTINGS)
    fields = _fields(problems)
    assert fields[:3] == ["businessArea", "subArea", "situation"]
    for f in ("examples", "checked", "checkpoint", "outcome", "owner"):
        assert f in fields
    # form order: The policy -> What happens -> Governance
    assert fields.index("situation") < fields.index("examples") < fields.index("outcome") < fields.index("owner")


def test_approve_needs_a_decider_and_a_positive_response_time():
    form = _ready_form()
    form["deciders"] = []
    form["responseTime"] = "PT0H"
    fields = _fields(R.activation_problems(form, REGISTRY, SETTINGS))
    assert "deciders" in fields and "responseTime" in fields


def test_notify_needs_someone_to_tell():
    form = _ready_form()
    form["outcome"] = "notify"
    form["notify"] = []
    assert "notify" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_limit_on_needs_text():
    form = _ready_form()
    form["limit"] = {"on": True, "text": "  "}
    assert "limit" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_flipped_example_blocks_active():
    form = _ready_form()
    form["examples"][0]["flipped"] = True
    assert "examples" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_unknown_tool_is_routed_to_an_administrator():
    form = _ready_form()
    form["hidden"]["condition"]["all"][0]["value"] = ["refund.isue"]
    msgs = [p for p in R.activation_problems(form, REGISTRY, SETTINGS) if p["field"] == "registry"]
    assert msgs and msgs[0]["code"] == "unknown_name" and msgs[0]["names"] == ["refund.isue"]
    assert msgs[0].get("routeTo") == "administrator"
    # the screen says "orchestrator" (from the code); the server's own words never do
    assert "orchestrator" not in msgs[0]["message"].lower()


def test_amount_without_currency_blocks_active():
    form = _ready_form()
    form["hidden"]["units"]["currency"] = None
    assert "units" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_approve_and_block_need_a_message_to_the_agent():
    form = _ready_form()
    form["messageForAgent"] = ""
    assert "messageForAgent" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_unavailable_input_cannot_be_enforced_yet():
    form = _ready_form()
    form["hidden"]["inputs"].append({"name": "refunds to this customer in the last 30 days", "field": "agg.refunds_30d",
                                     "type": "number", "from": "total:refunds_30d", "showApprover": False, "sensitive": False})
    he = R.how_enforced(form, REGISTRY, SETTINGS)
    assert he["ok"] is False
    assert he["cantEnforce"] == ["Can't be enforced yet: the orchestrator does not receive "
                                 "refunds to this customer in the last 30 days at this point"]
    assert he["cantEnforceNames"] == ["refunds to this customer in the last 30 days"]
    probs = [p for p in R.activation_problems(form, REGISTRY, SETTINGS) if p["field"] == "inputs"]
    assert len(probs) == 1 and probs[0]["code"] == "cant_enforce"
    assert probs[0]["missing"] == ["refunds to this customer in the last 30 days"]
    assert probs[0]["routeTo"] == "administrator" and "orchestrator" not in probs[0]["message"].lower()


def test_agent_reported_missing_input_also_cannot_be_enforced():
    form = _ready_form()
    form["hidden"]["missingInputs"] = [{"name": "customer's country", "reason": "no registry field"}]
    assert R.how_enforced(form, REGISTRY, SETTINGS)["ok"] is False


def test_how_enforced_reads_in_plain_words():
    he = R.how_enforced(_ready_form(), REGISTRY, SETTINGS)
    assert he == {"ok": True,
                  "checkedWhen": "the agent is about to do this: issuing a refund or credit (before a tool runs)",
                  "needsToKnow": "Refund amount (USD, from the action), Tool (from the action), Agent's reason (from the action)",
                  "then": "the action pauses and Finance Manager is asked to approve, then CFO if there is no answer "
                          "within 4 hours; if the last level does not answer, the action is rejected. The agent tells the person: \"Your refund needs a manager's approval. "
                          "You will hear back within 4 hours.\""}


def test_confirmation_is_cleared_by_situation_or_flip_but_not_by_owner():
    old = _ready_form()
    new = copy.deepcopy(old); new["situation"] += " Today."
    assert R.confirmation_cleared(old, new)
    new = copy.deepcopy(old); new["examples"][1]["flipped"] = True
    assert R.confirmation_cleared(old, new)
    new = copy.deepcopy(old); new["owner"] = "CFO"
    assert not R.confirmation_cleared(old, new)


def test_extraction_confidence_high_medium_low():
    form = _ready_form()
    form["hidden"]["setBy"] = "extraction_agent"
    for ex, agent in zip(form["examples"], ["approve", "none", "none", "none"]):
        ex["agentExpected"] = agent
    assert R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS) == {"level": "High", "failed": []}
    form["examples"][0]["agentExpected"] = "none"            # agent disagrees with code
    assert R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)["level"] == "Medium"
    form["source"]["excerpt"] = "Refunds above $500 need approval."   # not word for word
    low = R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)
    assert low["level"] == "Low"
    assert set(low["failed"]) == {"The excerpt does not appear word for word in the document",
                                  "The agent's expected result differs from the computed one for 1 example"}


def test_confidence_and_the_agent_note_use_one_word_for_word_check():
    """Minor 8: a short excerpt that does appear in the text is still not proof (min 4 words),
    for confidence exactly as for the extraction run's agent note."""
    from services.agent_policy import extraction_run
    form = _ready_form()
    for ex, agent in zip(form["examples"], ["approve", "none", "none", "none"]):
        ex["agentExpected"] = agent
    short = DOC_TEXT.split()[:3]
    form["source"]["excerpt"] = " ".join(short)
    assert " ".join(short) in DOC_TEXT
    assert R.excerpt_grounded(form["source"]["excerpt"], DOC_TEXT) is False
    assert "The excerpt does not appear word for word in the document" in \
        R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)["failed"]
    assert extraction_run.readiness.excerpt_grounded is R.excerpt_grounded


def test_no_source_means_no_extraction_confidence():
    form = _ready_form(); form["source"] = None
    assert R.extraction_confidence(form, None, REGISTRY, SETTINGS) is None


def _planned_in_condition(also_in_inputs):
    form = _ready_form()
    form["hidden"]["condition"]["all"].append({"field": "agg.refunds_30d", "op": "gt", "value": 3})
    if also_in_inputs:
        form["hidden"]["inputs"].append({"name": "refunds in 30 days", "field": "agg.refunds_30d",
                                         "type": "number", "from": "total:refunds_30d",
                                         "showApprover": False, "sensitive": False})
    return form


def test_condition_field_not_live_and_not_in_inputs_cannot_be_enforced():
    form = _planned_in_condition(also_in_inputs=False)
    assert "inputs" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))
    he = R.how_enforced(form, REGISTRY, SETTINGS)
    assert he["ok"] is False
    assert he["cantEnforce"] == ["Can't be enforced yet: the orchestrator does not receive "
                                 "refunds in 30 days at this point"]
    form["source"] = {"excerpt": "Refunds or credits above $500", "documentId": "d"}
    conf = R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)
    assert conf["level"] != "High"


def test_field_in_both_inputs_and_condition_is_reported_once():
    form = _planned_in_condition(also_in_inputs=True)
    assert len(R.how_enforced(form, REGISTRY, SETTINGS)["cantEnforce"]) == 1


def test_every_problem_has_a_stable_code_and_no_message_names_the_orchestrator():
    problems = R.activation_problems({"name": "", "outcome": "approve", "deciders": [], "responseTime": "PT0H",
                                      "limit": {"on": True, "text": ""}}, REGISTRY, SETTINGS)
    codes = {p["field"]: p["code"] for p in problems}
    assert codes["name"] == "name_required" and codes["businessArea"] == "business_area_required"
    assert codes["subArea"] == "sub_area_required" and codes["situation"] == "situation_required"
    assert codes["checkpoint"] == "checkpoint_required" and codes["examples"] == "examples_missing"
    assert codes["checked"] == "not_confirmed" and codes["deciders"] == "deciders_required"
    assert codes["responseTime"] == "response_time_invalid" and codes["messageForAgent"] == "message_for_agent_required"
    assert codes["limit"] == "limit_text_required" and codes["owner"] == "owner_required"
    assert all(p["code"] and p["message"] and "orchestrator" not in p["message"].lower() for p in problems)


@pytest.mark.parametrize("mutate, field, code", [
    (lambda f: f.update(outcome=None), "outcome", "outcome_required"),
    (lambda f: f.update(outcome="notify", notify=[]), "notify", "notify_required"),
    (lambda f: f["hidden"]["units"].update(currency=None), "units", "currency_required"),
    (lambda f: f["hidden"].update(timeWindow={"days": ["mon"], "from": "18:00", "to": "08:00", "timeZone": ""}),
     "timeWindow", "time_zone_required"),
    (lambda f: f["examples"][0].update(flipped=True), "examples", "example_flipped"),
])
def test_codes_for_the_remaining_problems(mutate, field, code):
    form = _ready_form()
    mutate(form)
    got = [p["code"] for p in R.activation_problems(form, REGISTRY, SETTINGS) if p["field"] == field]
    assert got == [code]
