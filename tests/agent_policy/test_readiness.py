import copy

from services.agent_policy import readiness as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

DOC_TEXT = "1.1 Refunds or credits above $500 need approval from the Finance Manager.\n1.2 ..."


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
    assert msgs and msgs[0]["message"].startswith("This policy refers to something the orchestrator does not recognise")
    assert msgs[0].get("routeTo") == "administrator"


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
    assert "inputs" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


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
                          "within 4 hours. The agent tells the person: \"Your refund needs a manager's approval. "
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
