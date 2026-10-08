import copy
import pytest

from services.agent_policy import conditions as C
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS


def test_translation_uses_one_evaluator():
    eng = C.to_engine(FORM_EXAMPLE["hidden"]["condition"])
    assert eng == {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                           {"field": "args.amount", "op": ">", "value": 500}]}


def test_unknown_operator_is_refused():
    with pytest.raises(C.ConditionError):
        C.to_engine({"field": "args.amount", "op": "greater", "value": 1})


def test_fields_and_tool_names():
    cond = FORM_EXAMPLE["hidden"]["condition"]
    assert C.condition_fields(cond) == {"tool.name", "args.amount"}
    assert C.tool_names(cond) == {"refund.issue", "credit.issue"}


@pytest.mark.parametrize("amount,expected", [(501, "approve"), (500, "none"), (499, "none")])
def test_boundary_results_are_computed_by_code(amount, expected):
    out = C.example_result(FORM_EXAMPLE["hidden"]["condition"], "approve",
                           {"tool.name": "refund.issue", "args.amount": amount})
    assert out == expected


def test_missing_input_fails_closed_by_default():
    out = C.example_result(FORM_EXAMPLE["hidden"]["condition"], "block", {"tool.name": "refund.issue"})
    assert out == "block"


def test_reviewer_view_labels_and_flip():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["examples"][1]["flipped"] = True
    rows = C.reviewer_view(form, SETTINGS)
    assert [r["label"] for r in rows] == ["A person decides", "Nothing happens", "Nothing happens", "Nothing happens"]
    assert rows[1]["flipped"] and rows[1]["reviewer_expects"] == "approve"
    assert rows[0]["reviewer_expects"] == "approve"
