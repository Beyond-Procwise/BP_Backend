import json
import re
from pathlib import Path

from services.agent_policy import conflict_payload as cp
from tests.agent_policy.test_conflict_detect import make

BRIEF = Path(__file__).resolve().parents[2] / "specs" / "2026-10-08-agent-policy-governance-brief.md"


def _sample():
    text = BRIEF.read_text()
    section = text[text.index("### 4.4 Conflict case payload"):]
    return json.loads(re.search(r"```json\n(.*?)\n```", section, re.S).group(1))


def _docs():
    a = make("FIN-0012", cond={"all": [{"field": "args.amount", "op": "gt", "value": 500}]})
    b = make("CUS-0004", outcome="block", source="Customer Refund Standard",
             cond={"all": [{"field": "args.amount", "op": "gt", "value": 10000}]})
    return a, b


def _build(kind="live", **kw):
    a, b = _docs()
    pols = [cp.policy_entry(a), cp.policy_entry(b)]
    args = dict(raised_at="2026-10-08T09:20:00Z", policies=pols,
                overlap_example={"tool": "refund.issue", "amount": 12400}, standing_rules=[],
                prior={"sameConflict": 2, "lastOutcome": "rejected"},
                options=cp.policy_options(a, b), respond_within="PT4H", on_timeout="reject",
                case="pc_7")
    if kind == "live":
        args["action"] = {"agent": "x", "tool": "refund.issue", "plain": "Issue a refund", "args": {"amount": 12400}}
    args.update(kw)
    return cp.build(kind, **args)


def test_build_matches_brief_shape():
    sample = _sample()
    live = _build("live")
    assert set(live) - {"summary"} == set(sample)
    assert set(live["priorDecisions"]) == set(sample["priorDecisions"])
    assert set(live["overlap"]) == set(sample["overlap"])
    assert set(live["action"]) == set(sample["action"])
    for got, want in zip(live["policies"], sample["policies"]):
        assert set(got) - {"deciders"} == set(want) - {"deciders"}
        assert set(got["source"]) == set(want["source"])
    assert live["schema"] == "policy-conflict/1" and live["caseId"] == "pc_7"
    pol = _build("policy")
    assert "action" not in pol and set(pol) == set(live) - {"action"}


def test_policy_entry_shape():
    a, b = _docs()
    ea, eb = cp.policy_entry(a), cp.policy_entry(b)
    assert ea["deciders"] == ["Finance Manager", "CFO"] and "deciders" not in eb
    assert ea["businessArea"] == " / ".join([a["businessArea"]["primary"], a["businessArea"]["subArea"]])
    assert ea["situation"] == a["trigger"]["plain"] and ea["id"] == "FIN-0012"


def test_live_args_only_condition_fields():
    a, b = _docs()
    got = cp.condition_args([a, b], {"amount": 12400, "currency": "USD", "customer_email": "x@y", "note": "n"})
    assert got == {"amount": 12400}


def test_condition_values():
    a, b = _docs()
    assert cp.condition_values([a, b], {"args.amount": 5, "args.note": "n", "tool.name": "t"}) == {"args.amount": 5}


def test_options_for_block_pair_offer_only_block_keep_both():
    a, b = _docs()
    assert cp.policy_options(a, b) == ["keep_both:CUS-0004", "change:FIN-0012", "change:CUS-0004",
                                       "limit:FIN-0012", "limit:CUS-0004", "retire:FIN-0012", "retire:CUS-0004"]
    a2 = make("A-1")
    b2 = make("B-1", deciders=("Legal",))
    assert cp.policy_options(a2, b2)[:3] == ["keep_both:A-1", "keep_both:B-1", "change:A-1"]
    assert len(cp.policy_options(a2, b2)) == 8


def test_scope_of():
    assert cp.scope_of("keep_both:A-1") == "standing_rule"
    assert cp.scope_of("retire:A-1") == "this_action"


def test_why_line_block_vs_approve():
    a, b = _docs()
    assert cp.why_line([a, b]) == ("One policy needs approval from Finance Manager then CFO; "
                                   "the other does not allow this at all.")
    a2, b2 = make("A-1"), make("B-1", deciders=("Legal",))
    assert cp.why_line([a2, b2]) == "The policies name different approvers: Finance Manager then CFO and Legal."
    assert cp.why_line([a2, a2]) == "The policies name different approvers: Finance Manager then CFO and Finance Manager then CFO."
    n = make("N-1", outcome="notify")
    assert cp.why_line([n, a2]) == "The policies say different things about this action."


def test_summary_keeps_excerpt_verbatim():
    a, b = _docs()
    odd = '  "Refunds"   above  “$500”\tneed\napproval.  '
    a["source"]["excerpt"] = odd
    p = cp.build("live", raised_at="t", policies=[cp.policy_entry(a), cp.policy_entry(b)],
                 overlap_example={}, standing_rules=[], prior={"sameConflict": 0, "lastOutcome": None},
                 options=[], respond_within=None, on_timeout="reject",
                 action={"plain": "p", "args": {}})
    assert p["policies"][0]["source"]["excerpt"] == odd
    assert p["summary"]["policies"][0]["excerpt"] == odd
    assert p["summary"]["why"]
    assert set(p["summary"]["policies"][0]) == {"id", "situation", "outcome", "owner", "excerpt"}


def test_case_id_round_trip():
    assert cp.case_id(12) == "pc_12" and cp.parse_case_id("pc_12") == 12
    assert cp.parse_case_id("12") == 12
    assert cp.parse_case_id("pc_x") is None and cp.parse_case_id("") is None and cp.parse_case_id("1a") is None


def test_to_columns_maps_design_section_5():
    live = _build("live")
    c = cp.to_columns(live)
    assert c["subject_type"] == "live_conflict" and cp.to_columns(_build("policy"))["subject_type"] == "policy_conflict"
    assert set(c["facts"]) == {"action", "policies", "standingRules", "priorDecisions", "summary", "schema"}
    assert c["evidence"] == [{"kind": "overlap", "example": live["overlap"]["example"]}]
    assert c["options"] == live["options"] and c["on_timeout"] == "reject"
    assert "action" not in cp.to_columns(_build("policy"))["facts"]


def test_returned_decision_maps_design_section_5():
    row = {"decision_id": 7, "decision": "keep_both:CUS-0004", "decision_scope": "standing_rule",
           "actioned_by": "cfo", "actioned_at": "2026-10-09T10:00:00Z", "override_reason": "ok"}
    assert cp.returned_decision(row) == {"caseId": "pc_7", "decision": "keep_both:CUS-0004", "scope": "standing_rule",
                                         "decidedBy": "cfo", "decidedAt": "2026-10-09T10:00:00Z", "reason": "ok"}
