import copy
import time

import pytest

from services.agent_policy import conditions, conflict_detect as cd
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS


def make(key="FIN-0001", *, outcome="approve", tools=("refund.issue", "credit.issue"), cond=None,
         source="Finance Payments Policy", deciders=("Finance Manager", "CFO"), checkpoint=None,
         examples=None):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["outcome"] = outcome
    form["source"]["document"] = source
    form["deciders"] = list(deciders) if outcome == "approve" else []
    form["notify"] = ["Someone"] if outcome == "notify" else []
    form["hidden"]["actions"]["tools"] = list(tools)
    if cond is not None:
        form["hidden"]["condition"] = cond
    if checkpoint:
        form["hidden"]["checkpoint"] = checkpoint
    if examples is not None:
        form["examples"] = examples
    return compile_policy(form, policy_key=key, version=1, status="live", settings=SETTINGS, never_suggest=False)


def amount(op, v):
    return {"field": "args.amount", "op": op, "value": v}


def both_match(a, b, w):
    for d in (a, b):
        assert cd._matches(d, cd._cond(d), w)


def test_finds_witness_between_approve_over_500_and_block_over_10000_from_other_source():
    a = make("FIN-0001", cond={"all": [amount("gt", 500)]})
    b = make("LEG-0001", outcome="block", source="Legal Policy", cond={"all": [amount("gt", 10000)]})
    w = cd.witness(a, b)
    assert w is not None and w["args.amount"] > 10000
    assert w["tool.name"] in cd._tools(a) and w["tool.name"] in cd._tools(b)
    both_match(a, b, w)


def test_no_witness_when_ranges_do_not_meet():
    a = make("A-1", cond={"all": [amount("gt", 500)]})
    b = make("B-1", outcome="block", cond={"all": [amount("lt", 400)]})
    assert cd.witness(a, b) is None


def test_no_witness_when_tool_lists_are_disjoint():
    a = make("A-1", tools=("refund.issue",), cond={"all": [amount("gt", 500)]})
    b = make("B-1", outcome="block", tools=("credit.issue",), cond={"all": [amount("gt", 500)]})
    assert cd.witness(a, b) is None


def test_witness_never_relies_on_missing_field():
    a = make("A-1", cond={"all": [amount("gt", 500)]})
    b = make("B-1", outcome="block", cond={"all": [{"field": "args.currency", "op": "eq", "value": "EUR"}]})
    w = cd.witness(a, b)
    assert w is not None and w["args.currency"] == "EUR" and w["args.amount"] > 500
    both_match(a, b, w)


def test_examples_tried_first():
    a = make("A-1", cond={"all": [amount("gt", 500)]})
    b = make("B-1", outcome="block", cond={"all": [amount("gt", 100)]})
    ex = {"tool.name": "credit.issue", "args.amount": 777}
    assert cd.witness(a, b, [ex]) == ex


def test_unreadable_condition_raises():
    a = make("A-1", cond={"all": [amount("gt", 500)]})
    b = make("B-1", outcome="block")
    b["trigger"]["condition"] = {"field": "args.amount", "op": "nonsense", "value": 1}
    with pytest.raises(conditions.ConditionError):
        cd.witness(a, b)


def test_witness_search_is_bounded():
    def wide(prefix):
        return {"all": [{"field": f"args.f{i}", "op": "in", "value": [f"{prefix}{i}-{n}" for n in range(200)]}
                        for i in range(3)]}
    a = make("A-1", cond=wide("a"))
    b = make("B-1", outcome="block", cond=wide("b"))
    t = time.perf_counter()
    assert cd.witness(a, b) is None
    assert time.perf_counter() - t < 2


def test_design_time_pair_table():
    base = make("A-1")
    other = make("B-1", outcome="block", source="Legal Policy")
    assert cd.design_time_pair(base, other) is True
    assert cd.design_time_pair(base, make("B-1", outcome="block")) is False                 # same source
    assert cd.design_time_pair(base, make("B-1", source="Legal Policy")) is False           # same outcome
    assert cd.design_time_pair(base, make("B-1", outcome="notify", source="Legal Policy")) is False
    assert cd.design_time_pair(make("B-1", outcome="notify", source="Legal Policy"), base) is False
    assert cd.design_time_pair(base, make("B-1", outcome="block", source="Legal Policy",
                                          checkpoint="agent.run.before")) is False
    assert cd.design_time_pair(base, make("A-1", outcome="block", source="Legal Policy")) is False  # same key


def test_helpers():
    assert cd.pair_key("B-1", "A-1") == cd.pair_key("A-1", "B-1") == "A-1|B-1"
    assert cd.pair_key("A-1", "A-1") == "A-1"
    a = make("A-1", source="  Finance PAYMENTS Policy ")
    assert cd.source_of(a) == "finance payments policy"
    assert cd.deciders_of(a) == ("Finance Manager", "CFO")
    assert cd.deciders_of(make("B-1", outcome="block")) == ()
    assert cd.deciders_differ(a, make("B-1", deciders=("CFO",)))
    assert not cd.deciders_differ(a, make("B-1"))
    assert conditions._leaves is conditions.leaves
