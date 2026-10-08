import copy

from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS


def _compile(form, **kw):
    args = dict(policy_key="FIN-0012", version=2, status="live", settings=SETTINGS, never_suggest=False)
    args.update(kw)
    return compile_policy(form, **args)


def test_compiles_the_brief_example_shape():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    doc = _compile(form)
    assert doc["schema"] == "hard-policy/2" and doc["id"] == "FIN-0012" and doc["status"] == "live"
    assert doc["businessArea"] == {"primary": "Finance", "subArea": "Refunds and credits"}
    assert doc["scope"] == {"agents": ["*"], "tools": ["*"], "skills": ["*"], "limit": None}
    assert doc["trigger"]["events"] == ["tool.call.before"]
    assert doc["trigger"]["onMissingData"] == "fail_closed"
    assert doc["trigger"]["checkedBy"] == "user_8841"
    assert doc["enforcement"] == {"outcome": "approve", "intervention": {
        "escalateTo": [{"type": "role", "name": "Finance Manager"}, {"type": "role", "name": "CFO"}],
        "sla": {"source": "company_default", "respondWithin": "PT4H", "onTimeout": "escalate_next"}}}
    assert doc["outputs"]["toAgent"] == {"onMatch": "paused_for_approval", "reasonCode": "FIN-0012.over_limit",
                                         "reason": "Refunds over $500 need Finance approval.",
                                         "whilePaused": "no_retry",
                                         "messageForPerson": form["messageForPerson"]}
    assert doc["outputs"]["toApprover"]["show"] == ["args.amount", "agent.reason"]
    assert doc["outputs"]["audit"] == {"logInputs": True, "mask": []}
    assert doc["learning"] == {"eligible": True}
    assert doc["conflicts"] == []


def test_single_level_times_out_to_reject():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["deciders"] = ["Finance Manager"]
    assert _compile(form)["enforcement"]["intervention"]["sla"]["onTimeout"] == "reject"


def test_override_response_time_is_stored_as_iso_duration():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["responseTime"] = "PT6H"
    sla = _compile(form)["enforcement"]["intervention"]["sla"]
    assert sla == {"source": "policy", "respondWithin": "PT6H", "onTimeout": "escalate_next"}


def test_switching_outcome_drops_previous_fields_from_json():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["outcome"] = "block"          # deciders + responseTime still in form state on purpose
    form["responseTime"] = "PT6H"
    doc = _compile(form)
    assert doc["enforcement"] == {"outcome": "block"}
    assert doc["outputs"]["toApprover"] is None
    assert "whilePaused" not in doc["outputs"]["toAgent"]
    assert doc["outputs"]["toAgent"]["onMatch"] == "blocked"
    assert doc["learning"] == {"eligible": False}
    form["outcome"] = "notify"
    form["notify"] = ["Finance Manager"]
    doc = _compile(form)
    assert doc["enforcement"] == {"outcome": "notify", "notify": [{"type": "role", "name": "Finance Manager"}]}
    assert doc["outputs"]["toAgent"]["onMatch"] == "allowed"
    assert doc["outputs"]["toNotify"] == {"to": ["Finance Manager"]}


def test_never_suggest_turns_learning_off():
    assert _compile(copy.deepcopy(FORM_EXAMPLE), never_suggest=True)["learning"] == {"eligible": False}


def test_limit_text_and_sensitive_mask():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["limit"] = {"on": True, "text": "EU support agents only"}
    form["hidden"]["inputs"][2]["sensitive"] = True
    doc = _compile(form)
    assert doc["scope"]["limit"] == "EU support agents only"
    assert doc["outputs"]["audit"]["mask"] == ["agent.reason"]


def test_compile_is_pure():
    form = copy.deepcopy(FORM_EXAMPLE)
    before = copy.deepcopy(form)
    assert _compile(form) == _compile(form)
    assert form == before
