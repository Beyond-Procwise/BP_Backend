"""A policy's ``condition`` decides whether the policy applies to this request.

The gate used to match on action name alone, so "refunds over $500" could only
be written as "refunds", and $499 and $501 got the same answer. These tests pin
the three outcomes a condition can have -- applies, does not apply, cannot be
told -- and that "cannot be told" never quietly becomes "does not apply".
"""

import pytest

from src.services import guardrail, policy_condition
from src.services.policy_condition import ConditionError, MissingField
from tests.guardrails.test_guardrail_gate import GateEngine, engine_with, approver


def over_500():
    return {"field": "amount", "op": ">", "value": 500}


# --- the evaluator ---------------------------------------------------------


@pytest.mark.parametrize(
    "condition,context,expected",
    [
        ({"field": "amount", "op": ">", "value": 500}, {"amount": 501}, True),
        ({"field": "amount", "op": ">", "value": 500}, {"amount": 500}, False),
        ({"field": "amount", "op": ">=", "value": 500}, {"amount": "500.00"}, True),
        ({"field": "amount", "op": "<", "value": 200}, {"amount": 199.99}, True),
        ({"field": "role", "op": "==", "value": "VIP"}, {"role": "VIP"}, True),
        ({"field": "role", "op": "!=", "value": "VIP"}, {"role": "VIP"}, False),
        ({"field": "region", "op": "in", "value": ["EU", "UK"]}, {"region": "UK"}, True),
        ({"field": "region", "op": "not_in", "value": ["EU"]}, {"region": "US"}, True),
        ({"field": "order.age_days", "op": ">", "value": 90}, {"order": {"age_days": 91}}, True),
        ({"field": "note", "op": "exists"}, {"note": "x"}, True),
        ({"field": "note", "op": "exists"}, {}, False),
        ({"not": over_500()}, {"amount": 10}, True),
        ({"all": [over_500(), {"field": "role", "op": "==", "value": "VIP"}]},
         {"amount": 600, "role": "std"}, False),
        # 10pm-6am wraps midnight, so it is written as an `any`
        ({"any": [{"field": "hour", "op": ">=", "value": 22},
                  {"field": "hour", "op": "<", "value": 6}]}, {"hour": 23}, True),
        ({"any": [{"field": "hour", "op": ">=", "value": 22},
                  {"field": "hour", "op": "<", "value": 6}]}, {"hour": 12}, False),
    ],
)
def test_evaluates(condition, context, expected):
    assert policy_condition.evaluate(condition, context) is expected


def test_a_missing_field_is_not_a_false():
    with pytest.raises(MissingField):
        policy_condition.evaluate(over_500(), {})


def test_a_missing_field_inside_any_is_still_missing_unless_another_branch_is_true():
    cond = {"any": [over_500(), {"field": "role", "op": "==", "value": "VIP"}]}
    assert policy_condition.evaluate(cond, {"role": "VIP"}) is True
    with pytest.raises(MissingField):
        policy_condition.evaluate(cond, {"role": "std"})


def test_a_missing_field_inside_all_is_false_when_another_branch_is_false():
    cond = {"all": [over_500(), {"field": "role", "op": "==", "value": "VIP"}]}
    assert policy_condition.evaluate(cond, {"role": "std"}) is False


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "amount > 500",
        {},
        {"field": "amount", "op": "~", "value": 1},
        {"field": "amount", "op": ">"},
        {"op": ">", "value": 1},
        {"all": []},
        {"all": "nope"},
        {"field": "amount", "op": "in", "value": 5},
        {"all": [over_500()], "any": [over_500()]},
    ],
)
def test_a_malformed_condition_raises(bad):
    with pytest.raises(ConditionError):
        policy_condition.evaluate(bad, {"amount": 1})


def test_comparing_a_number_to_text_raises_rather_than_guessing():
    with pytest.raises(ConditionError):
        policy_condition.evaluate(over_500(), {"amount": "lots"})


def test_booleans_are_not_numbers():
    with pytest.raises(ConditionError):
        policy_condition.evaluate(over_500(), {"amount": True})


# --- the gate --------------------------------------------------------------


def refund_policy(effect, condition=None, pid="P1"):
    details = {
        "policy_identifier": pid,
        "required_role": "Approver",
        "applies_to": ["refund.issue"],
        "rules": {"effect": effect, "reason": "refund over limit"},
    }
    if condition is not None:
        details["condition"] = condition
    return {"policyId": pid, "policyName": pid, "details": details, "raw_row": {"version": 3}}


def decide(engine, context):
    return guardrail._evaluate(
        "refund.issue", "transact", approver(), context, policy_engine=engine
    )


def test_a_deny_with_a_condition_denies_only_when_the_condition_holds():
    engine = engine_with(
        refund_policy("deny", over_500()),
        refund_policy("allow", pid="P0"),
    )
    assert decide(engine, {"amount": 499}).allowed is True
    over = decide(engine, {"amount": 501})
    assert over.allowed is False
    assert over.policy_id == "P1"
    assert over.policy_version == 3


def test_a_deny_whose_condition_cannot_be_checked_is_not_skipped():
    engine = engine_with(refund_policy("deny", over_500()), refund_policy("allow", pid="P0"))
    for context in ({}, None, {"currency": "USD"}):
        d = decide(engine, context)
        assert d.allowed is False
        assert d.unresolved is True
        assert d.policy_id == "P1"


def test_a_malformed_condition_denies_it_does_not_evaporate():
    engine = engine_with(refund_policy("deny", {"field": "amount", "op": "~", "value": 1}))
    d = decide(engine, {"amount": 1})
    assert d.allowed is False
    assert d.unresolved is False
    assert d.policy_id == "P1"


def test_a_conditional_allow_that_does_not_apply_permits_nothing():
    engine = engine_with(refund_policy("allow", {"field": "amount", "op": "<=", "value": 200}))
    assert decide(engine, {"amount": 150}).allowed is True
    d = decide(engine, {"amount": 250})
    assert d.allowed is False
    assert d.unresolved is True  # transact is irreversible; nobody permitted it


def test_a_policy_that_does_not_apply_does_not_enforce_its_required_role():
    policy = refund_policy("allow", {"field": "amount", "op": "<=", "value": 200})
    policy["details"]["required_role"] = "Admin"
    other = refund_policy("allow", pid="P0")
    d = decide(engine_with(policy, other), {"amount": 500})
    assert d.allowed is True
    assert d.policy_id == "P0"


def test_a_policy_with_no_condition_behaves_exactly_as_before():
    engine = engine_with(refund_policy("allow"))
    assert decide(engine, {}).allowed is True
    assert decide(engine, None).allowed is True
