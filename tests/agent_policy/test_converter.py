import re

from services.agent_policy import contract, readiness, registry
from services.agent_policy.compiler import compile_policy
from services.agent_policy.converter import to_form
from services.agent_policy.extraction_schema import ProposedPolicy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

TAXONOMY = [{"areaName": "Finance", "subAreas": ["General", "Refunds and credits", "Payments"]},
            {"areaName": "Unassigned", "subAreas": ["General"]}]
EXCERPT = ("Refunds or credits above $500 need approval from the Finance Manager. "
           "Refunds above $10,000 are not allowed.")


def _proposal(**over):
    base = dict(
        name="Refund or credit over $500", category="Approval",
        business_area="Finance", sub_area="Refunds and credits",
        situation="The agent is about to issue a refund or credit above $500.",
        match="all", rules=[{"field": "args.amount", "op": "gt", "value_number": 500}],
        outcome="approve", outcome_phrase="need approval from", deciders=["Finance Manager"],
        reference="1.1", excerpt=EXCERPT,
        examples=[{"values": [{"field": "args.amount", "value_number": 501}], "expected": "approve"},
                  {"values": [{"field": "args.amount", "value_number": 500}], "expected": "none"},
                  {"values": [{"field": "tool.name", "value_text": "supplier_ranking"},
                              {"field": "args.amount", "value_number": 900}], "expected": "none"}],
        checkpoint="tool.call.before", action_tools=["refund.issue", "credit.issue"],
        action_plain="issuing a refund or credit", currency="USD", amounts_include_tax=True,
        inputs=[{"name": "Refund amount", "field": "args.amount", "type": "number", "is_amount": True,
                 "unit": "USD"}],
        reason_code="over_limit", message_for_agent="Refunds over $500 need Finance approval.",
        message_for_person="Your refund needs a manager's approval.", owner="Chief Financial Officer")
    base.update(over)
    return ProposedPolicy.model_validate(base)


def _form(p=None, reg=REGISTRY):
    return to_form(p or _proposal(), document_title="Finance Payments Policy", document_version=1,
                   registry=reg, taxonomy=TAXONOMY)


def test_form_has_stage_one_keys_and_is_a_proposal():
    form = _form()
    assert set(form) == set(FORM_EXAMPLE)
    assert set(FORM_EXAMPLE["hidden"]) <= set(form["hidden"])
    assert form["checked"] is None
    assert form["hidden"]["setBy"] == "extraction_agent"
    assert form["responseTime"] is None and form["limit"] == {"on": False, "text": ""}
    assert form["source"] == {"document": "Finance Payments Policy", "documentVersion": 1,
                              "reference": "1.1", "excerpt": EXCERPT}
    assert form["outcomeBecause"] == "need approval from"
    assert form["hidden"]["condition"] == {"all": [
        {"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
        {"field": "args.amount", "op": "gt", "value": 500}]}
    fields = [i["field"] for i in form["hidden"]["inputs"]]
    assert {"tool.name", "agent.reason", "args.amount"} <= set(fields)
    assert form["hidden"]["units"] == {"currency": "USD", "convertOther": "rate_on_action_date",
                                       "amountsIncludeTax": True}
    assert form["hidden"]["timeWindow"] is None
    assert form["hidden"]["unknownNames"] == [] and form["hidden"]["agentNotes"] == []
    assert form["examples"][0] == {"input": {"tool.name": "refund.issue", "args.amount": 501},
                                   "agentExpected": "approve", "flipped": False}
    # an example that names its own tool keeps it
    assert form["examples"][2]["input"]["tool.name"] == "supplier_ranking"


def test_value_mapping_list_text_and_exists():
    p = _proposal(match="any", action_tools=[], rules=[
        {"field": "args.currency", "op": "in", "value_list": ["EUR", "GBP"]},
        {"field": "agent.name", "op": "eq", "value_text": "refund_bot"},
        {"field": "agent.reason", "op": "exists"}])
    cond = _form(p)["hidden"]["condition"]
    assert cond == {"any": [{"field": "args.currency", "op": "in", "value": ["EUR", "GBP"]},
                            {"field": "agent.name", "op": "eq", "value": "refund_bot"},
                            {"field": "agent.reason", "op": "exists"}]}


def test_a_tool_rule_suppresses_the_added_tool_leaf():
    p = _proposal(rules=[{"field": "tool.name", "op": "in", "value_list": ["refund.issue"]},
                         {"field": "args.amount", "op": "gt", "value_number": 500}])
    leaves = _form(p)["hidden"]["condition"]["all"]
    assert [l["field"] for l in leaves].count("tool.name") == 1


def test_invented_tool_name_goes_to_unknown_names():
    p = _proposal(action_tools=["refund.issue", "refund.mass_issue"],
                  rules=[{"field": "args.refund_total", "op": "gt", "value_number": 500}],
                  unknown_names=["refund approval queue"])
    unknown = _form(p)["hidden"]["unknownNames"]
    # the model's own list, plus what the code found itself: never trusted, always re-checked
    assert unknown == ["refund approval queue", "args.refund_total", "refund.mass_issue"]


def test_tiered_clause_gives_two_forms_with_one_reference():
    approve = _proposal()
    block = _proposal(name="Refund over $10,000", outcome="block", outcome_phrase="are not allowed",
                      deciders=[], action_tools=["refund.issue"],
                      rules=[{"field": "args.amount", "op": "gt", "value_number": 10000}])
    forms = [_form(approve), _form(block)]
    assert [f["outcome"] for f in forms] == ["approve", "block"]
    assert forms[0]["source"]["reference"] == forms[1]["source"]["reference"] == "1.1"
    assert forms[0]["source"]["excerpt"] == forms[1]["source"]["excerpt"] == EXCERPT


def test_off_taxonomy_area_becomes_none_with_a_note():
    form = _form(_proposal(business_area="Treasury", sub_area="Refunds and credits"))
    assert form["businessArea"] is None and form["subArea"] is None
    assert "The agent proposed business area 'Treasury', which is not in the taxonomy." in form["hidden"]["agentNotes"]


def test_off_taxonomy_sub_area_becomes_none_with_a_note():
    form = _form(_proposal(sub_area="Chargebacks"))
    assert form["businessArea"] == "Finance" and form["subArea"] is None
    assert any("'Chargebacks'" in n for n in form["hidden"]["agentNotes"])


def test_lookup_input_is_unregistered_and_cant_be_enforced():
    p = _proposal(inputs=[
        {"name": "Refund amount", "field": "args.amount", "type": "number", "is_amount": True},
        {"name": "Customer's refunds this month", "field": "agg.customer_refunds", "type": "number",
         "source": "lookup"}])
    form = _form(p)
    row = next(i for i in form["hidden"]["inputs"] if i["field"] == "agg.customer_refunds")
    assert row["from"] == "lookup:unregistered"
    totals = _form(_proposal(inputs=[{"name": "Refunds in 30 days", "field": "agg.r30", "type": "number",
                                      "source": "total"}]))
    assert any(i["from"] == "total:unregistered" for i in totals["hidden"]["inputs"])
    how = readiness.how_enforced(form, REGISTRY, SETTINGS)
    assert how["ok"] is False
    assert any(c.startswith("Can't be enforced yet") and "Customer's refunds this month" in c
               for c in how["cantEnforce"])


def test_time_window_set_when_either_end_is():
    form = _form(_proposal(time_window_from="18:00", time_zone="Europe/London"))
    assert form["hidden"]["timeWindow"] == {"from": "18:00", "to": None, "timeZone": "Europe/London"}


def test_condition_field_the_agent_forgot_to_list_is_added_as_an_input():
    p = _proposal(inputs=[], rules=[{"field": "args.amount", "op": "gt", "value_number": 500}])
    row = next(i for i in _form(p)["hidden"]["inputs"] if i["field"] == "args.amount")
    assert row["type"] == "number" and row["from"] == "action"


def _compiled(form):
    return compile_policy(form, policy_key="FIN-0001", version=1, status="draft",
                          settings=SETTINGS, never_suggest=False)


def test_converted_form_compiles_and_validates():
    assert contract.validate(_compiled(_form()), REGISTRY) == []


_REGISTRY_PROBLEM = re.compile(r"is not something the orchestrator recognises|is not available at")


def test_known_bad_registry_gives_only_registry_problems():
    bad = registry.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs",
         "status": "live"},
        {"kind": "action", "name": "refund.issue", "checkpoint": "tool.call.before", "plain": "x", "status": "live"},
        {"kind": "input", "name": "tool.name", "checkpoint": "tool.call.before", "plain": "tool",
         "value_type": "string", "source": "action", "status": "live"},
        {"kind": "input", "name": "args.amount", "checkpoint": "tool.call.before", "plain": "amount",
         "value_type": "number", "source": "action", "status": "planned"},
    ])
    problems = contract.validate(_compiled(_form(reg=bad)), bad)
    assert problems, "a registry missing credit.issue and agent.reason must be reported"
    assert all(_REGISTRY_PROBLEM.search(p) for p in problems), problems
    assert any("credit.issue" in p for p in problems)
    assert any("agent.reason" in p for p in problems)
