"""Live-conflict classification (pure): no DB, no model, no clock."""
from services.agent_policy.conflict_engine import LiveConflict, classify
from services.agent_policy.enforcement import Verdict


def hit(pid, outcome="approve", deciders=("Finance",), source=None, conflicts=None, unreadable=False):
    policy = {
        "id": pid,
        "enforcement": {
            "outcome": outcome,
            "intervention": {"escalateTo": [{"name": d} for d in deciders]} if outcome == "approve" else {},
        },
    }
    if source:
        policy["source"] = {"document": source}
    if conflicts:
        policy["conflicts"] = conflicts
    h = {"id": pid, "outcome": outcome, "policy": policy}
    if unreadable:
        h["unreadable"] = True
    return h


def verdict(*hits):
    v = Verdict()
    for h in hits:
        {"block": v.blocks, "approve": v.approvals, "notify": v.notifies}[h["outcome"]].append(h)
    return v


def rule(other, prevails):
    return {"with": other, "rule": "r", "caseId": 1, "decidedAt": "2026-10-09T00:00:00Z", "prevails": prevails}


def test_single_policy_is_no_conflict():
    assert classify(verdict(hit("A-0001"))) == LiveConflict()
    assert classify(verdict(hit("A-0001"))).kind is None


def test_same_source_tiers_with_different_deciders_are_not_a_conflict():
    a = hit("A-0001", deciders=("Finance",), source="Policy.pdf")
    b = hit("A-0002", deciders=("CFO",), source="policy.pdf")
    assert classify(verdict(a, b)).kind is None


def test_same_deciders_different_sources_apply_normally():
    a = hit("A-0001", source="One.pdf")
    b = hit("A-0002", source="Two.pdf")
    assert classify(verdict(a, b)).kind is None


def test_block_plus_approve_is_a_block_record():
    blk = hit("B-0001", "block", source="One.pdf")
    app = hit("A-0001", source="Two.pdf")
    lc = classify(verdict(blk, app))
    assert lc.kind == "block_record"
    assert lc.pairs == [("A-0001", "B-0001")]
    assert {h["id"] for h in lc.involved} == {"A-0001", "B-0001"}


def test_standing_rule_never_beats_a_block():
    blk = hit("B-0001", "block", source="One.pdf")
    app = hit("A-0001", source="Two.pdf", conflicts=[rule("B-0001", "A-0001")])
    assert classify(verdict(blk, app)).kind == "block_record"


def test_unreadable_block_with_approve_is_not_a_conflict():
    blk = hit("B-0001", "block", source="One.pdf", unreadable=True)
    app = hit("A-0001", source="Two.pdf")
    assert classify(verdict(blk, app)).kind is None


def test_two_approves_with_different_deciders_go_to_a_human():
    a = hit("A-0001", deciders=("Finance",), source="One.pdf")
    b = hit("A-0002", deciders=("Legal",), source="Two.pdf")
    lc = classify(verdict(a, b))
    assert lc.kind == "human"
    assert lc.last_level_only == {"A-0001", "A-0002"}
    assert [h["id"] for h in lc.required] == ["A-0001", "A-0002"]


def test_standing_rule_decides_automatically():
    a = hit("A-0001", deciders=("Finance",), source="One.pdf", conflicts=[rule("A-0002", "A-0001")])
    b = hit("A-0002", deciders=("Legal",), source="Two.pdf")
    lc = classify(verdict(a, b))
    assert lc.kind == "auto"
    assert [h["id"] for h in lc.required] == ["A-0001"]
    assert lc.rules == [rule("A-0002", "A-0001")]
    assert lc.last_level_only == set()


def test_partial_rule_set_over_three_policies_goes_to_a_human():
    a = hit("A-0001", deciders=("Finance",), source="One.pdf", conflicts=[rule("A-0002", "A-0001")])
    b = hit("A-0002", deciders=("Legal",), source="Two.pdf")
    c = hit("A-0003", deciders=("CFO",), source="Three.pdf")
    assert classify(verdict(a, b, c)).kind == "human"


def test_circular_rule_set_goes_to_a_human():
    a = hit("A-0001", deciders=("Finance",), source="One.pdf", conflicts=[rule("A-0002", "A-0001")])
    b = hit("A-0002", deciders=("Legal",), source="Two.pdf", conflicts=[rule("A-0003", "A-0002")])
    c = hit("A-0003", deciders=("CFO",), source="Three.pdf", conflicts=[rule("A-0001", "A-0003")])
    assert classify(verdict(a, b, c)).kind == "human"


def test_notify_hits_never_take_part():
    n = hit("N-0001", "notify", source="One.pdf")
    app = hit("A-0001", source="Two.pdf")
    assert classify(verdict(n, app)).kind is None
    blk = hit("B-0001", "block", source="Three.pdf")
    lc = classify(verdict(n, blk))
    assert lc.kind is None
