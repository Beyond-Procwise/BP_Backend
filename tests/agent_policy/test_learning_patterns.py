"""Which parts of a condition can be loosened and what the evidence supports. No database."""
from datetime import datetime, timedelta, timezone

from services.agent_policy import learning_patterns as LP
from services.agent_policy import learning_stats as LS
from services.agent_policy.enforcement import MASK
from services.agent_policy.settings import DEFAULTS

CFG = DEFAULTS["learning"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
COND = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                {"field": "args.amount", "op": "gt", "value": 500}]}


def decided(values, approved=True, n_people=3, days=40, field="args.amount"):
    """One decision per value, spread over `days`, by `n_people` people, 60 s each."""
    out, n = [], len(values)
    for i, v in enumerate(values):
        asked = T0 + timedelta(days=days * i / max(n - 1, 1))
        ok = approved if isinstance(approved, bool) else approved[i]
        out.append(LS.Decided(firing_id=i, values={field: v, "tool.name": "refund.issue"}, approved=ok,
                              decided_by=f"p{i % n_people}", asked_at=asked, decided_at=asked + timedelta(seconds=60)))
    return out


def test_eligible_leaves_are_thresholds_and_lists_every_match_must_satisfy():
    cond = {"all": [COND["all"][0], COND["all"][1],
                    {"field": "customer.country", "op": "in", "value": ["FR", "DE"]},
                    {"any": [{"field": "args.qty", "op": "gt", "value": 5}]},
                    {"not": {"field": "args.x", "op": "lt", "value": 1}},
                    {"field": "args.flag", "op": "eq", "value": True},
                    {"field": "args.one", "op": "in", "value": ["only"]}]}
    assert [(l.kind, l.field) for l in LP.eligible_leaves(cond)] == [
        ("threshold", "args.amount"), ("list", "customer.country")]


def test_a_single_leaf_condition_is_eligible():
    assert [l.field for l in LP.eligible_leaves({"field": "args.amount", "op": "gte", "value": 500})] == ["args.amount"]


def test_a_boolean_or_text_threshold_is_not_a_threshold():
    assert LP.eligible_leaves({"field": "a", "op": "gt", "value": True}) == []
    assert LP.eligible_leaves({"field": "a", "op": "gt", "value": "high"}) == []


def test_raise_the_limit_to_what_people_approved_rounded_inside_the_evidence():
    values = [510 + 10 * i for i in range(33)] + [837.42]          # 510 … 830, then 837.42
    [p] = LP.propose(COND, decided(values), CFG)
    assert (p["kind"], p["field"], p["op"], p["from"], p["to"]) == ("threshold", "args.amount", "gt", 500, 830)
    assert p["changeKey"] == "threshold:args.amount:gt:500"
    assert p["stats"]["decisions"] == 33 and p["to"] <= max(values)


def test_never_beyond_the_evidence_when_rounding_would_drop_below_the_guardrails():
    values = [801 + i * 0.1 for i in range(30)]                    # all between 801 and 803.9
    [p] = LP.propose(COND, decided(values), CFG)
    assert p["to"] == 803.9, "800 would leave no evidence; the furthest decided value is used"


def test_a_rejection_in_the_range_stops_the_suggestion_below_it():
    values = sorted([510 + 5 * i for i in range(40)] + [650.5] * 1)
    approved = [v != 650.5 for v in values]
    props = LP.propose(COND, decided(values, approved=approved), CFG)
    # 28 decisions sit below 650.5 (too few); every band that includes the rejection has a
    # Wilson bound under 0.85 until far more approvals pile up. With 41 decisions: 40/41 → 0.874.
    assert props == [] or all(p["stats"]["wilsonLower"] >= 0.85 for p in props)
    assert LP.propose(COND, decided(values[:30], approved=approved[:30]), CFG) == []


def test_too_little_evidence_proposes_nothing():
    assert LP.propose(COND, decided([600] * 29), CFG) == []


def test_masked_and_missing_values_are_not_evidence():
    ds = decided([MASK] * 40) + decided([None] * 5)
    assert LP.propose(COND, ds, CFG) == []


def test_text_numbers_are_read_as_the_evaluator_reads_them():
    [p] = LP.propose(COND, decided([str(600 + i) for i in range(30)]), CFG)
    assert p["to"] == 629, "rounding to 620 would leave 21 decisions, so the furthest decided value stands"


def test_a_lower_bound_is_loosened_downward():
    cond = {"field": "args.margin", "op": "lt", "value": 10}
    [p] = LP.propose(cond, decided([9.5 - 0.1 * i for i in range(30)], field="args.margin"), CFG)
    assert p["op"] == "lt" and p["from"] == 10 and 6.6 <= p["to"] < 10


def test_a_list_value_people_always_approve():
    cond = {"all": [{"field": "customer.country", "op": "in", "value": ["FR", "DE"]}]}
    ds = decided(["FR"] * 30, field="customer.country") + decided(["DE"] * 5, field="customer.country")
    [p] = LP.propose(cond, ds, CFG)
    assert (p["kind"], p["remove"], p["changeKey"]) == ("list", "FR", "list:customer.country:FR")


def test_targets_and_applied():
    p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830}
    looser = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 700}]}
    tighter = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 400}]}
    gone = {"all": [COND["all"][0]]}
    assert LP.targets(p, COND) and not LP.targets(p, looser)
    assert LP.applied(p, looser) and not LP.applied(p, COND) and not LP.applied(p, tighter)
    assert not LP.applied(p, gone), "dropping the limit makes the policy match MORE, which is not a loosening"
    q = {"kind": "list", "field": "c", "op": "in", "remove": "FR"}
    has = {"field": "c", "op": "in", "value": ["FR", "DE"]}
    assert LP.targets(q, has) and not LP.applied(q, has)
    assert LP.applied(q, {"field": "c", "op": "in", "value": ["DE"]})


def test_group_for_is_the_band_the_proposal_is_about():
    p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 600}
    assert [d.values["args.amount"] for d in LP.group_for(p, decided([550, 600, 601]))] == [550, 600]
