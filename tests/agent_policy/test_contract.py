import copy

from services.agent_policy import contract
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS


def _doc(mutate=None, status="live"):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    if mutate:
        mutate(form)
    return compile_policy(form, policy_key="FIN-0012", version=1, status=status,
                          settings=SETTINGS, never_suggest=False)


def test_the_example_is_valid():
    assert contract.validate(_doc(), REGISTRY) == []


def test_condition_field_missing_from_inputs_fails():
    def drop(form):
        form["hidden"]["inputs"] = [i for i in form["hidden"]["inputs"] if i["field"] != "args.amount"]
    problems = contract.validate(_doc(drop), REGISTRY)
    assert any("args.amount" in p and "inputs" in p for p in problems)


def test_live_without_checked_by_is_refused():
    problems = contract.validate(_doc(lambda f: f.update(checked=None)), REGISTRY)
    assert any("checkedBy" in p for p in problems)


def test_unknown_tool_is_refused():
    problems = contract.validate(_doc(lambda f: f["hidden"]["condition"]["all"][0].update(value=["refund.isue"])), REGISTRY)
    assert any("refund.isue" in p for p in problems)


def test_input_not_available_at_checkpoint_is_refused():
    def planned(form):
        form["hidden"]["inputs"].append({"name": "Refunds in 30 days", "field": "agg.refunds_30d", "type": "number",
                                         "from": "total:refunds_30d", "showApprover": False, "sensitive": False})
    problems = contract.validate(_doc(planned), REGISTRY)
    assert any("agg.refunds_30d" in p for p in problems)


def test_events_must_equal_checkpoint():
    doc = _doc()
    doc["trigger"]["events"] = ["message.send.before"]
    assert any("events" in p for p in contract.validate(doc, REGISTRY))


def test_block_with_intervention_fails_schema():
    doc = _doc(lambda f: f.update(outcome="block"))
    doc["enforcement"]["intervention"] = {"escalateTo": [], "sla": {}}
    assert contract.validate(doc, REGISTRY)


def test_notify_without_notify_fails_schema():
    doc = _doc(lambda f: f.update(outcome="notify", notify=[]))
    assert contract.validate(doc, REGISTRY)


def test_malformed_condition_is_a_problem_not_a_crash():
    doc = _doc()
    doc["trigger"]["condition"] = {"all": 5}
    problems = contract.validate(doc, REGISTRY)
    assert any("trigger.condition" in p for p in problems)


def test_effective_from_must_be_a_real_date():
    doc = _doc()
    doc["effective"]["from"] = "not-a-date"
    assert any(p.startswith("effective/from") for p in contract.validate(doc, REGISTRY))
    doc["effective"]["from"] = "2026-02-01"
    assert contract.validate(doc, REGISTRY) == []
