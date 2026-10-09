"""A live clash decided on precedent at the gate (design §3.2). Needs PROCWISE_TEST_LIVE_DB=1.

Policies are injected into the gate (tool tst_<hex>, keys TST-<hex><letter>), so no live policy is
written to the shared database. N comes from the governed limit (fixtures.precedent_n). Every
case, conflict and notification row made here is removed by LG.cleanup; firing rows stay.
"""
import copy
import os

import pytest

from engines import decision_engine as DE
from services.agent_policy import approvals as A
from services.agent_policy import conflict_live as CL
from services.agent_policy import gate as G
from services.agent_policy import replay as R
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.fixtures import precedent_n, precedent_range
from tests.agent_policy.test_conflict_live_gate import conn, ran, world  # noqa: F401 - fixtures

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
OWNER = "Chief Financial Officer"          # FORM_EXAMPLE's owner, on every test policy


def call(monkeypatch, w, docs, ran, *, wf=None, user_id="req@example.test"):
    """One scripted agent call (amount 900), in its own workflow unless `wf` repeats one."""
    if wf is None:
        wf = f"{w.wf}-{len(w.wfs)}"
        w.wfs.append(wf)
    LG.use(monkeypatch, docs, w, ran)
    res, _ = LG.run(monkeypatch, LG.stub_tools(w, ran), [LG._round(w.tool), LG.FINAL], workflow_id=wf,
                    user_id=user_id)
    return wf, res.calls[0].result


def members_of(conn, wf):
    return LG.rows(conn, "SELECT decision_id, policy_name FROM proc.bp_decision WHERE subject_type = %s "
                         "AND workflow_id = %s AND decision = 'approve_or_reject' ORDER BY decision_id",
                   (A.SUBJECT_TYPE, wf))


def live_of(conn, wf):
    return LG.rows(conn, "SELECT d.decision_id, d.decision, d.status, d.actioned_by, d.decision_scope, d.facts, "
                         "d.evidence, c.is_open, c.outcome, c.by_person FROM proc.bp_decision d "
                         "JOIN proc.bp_agent_policy_conflict c USING (decision_id) "
                         "WHERE d.subject_type = %s AND d.workflow_id = %s ORDER BY d.decision_id",
                   (CL.SUBJECT_LIVE, wf))


def firings_of(conn, wf):
    return LG.rows(conn, "SELECT firing_id, policy_key, result, decision_id, reason FROM proc.bp_policy_firing "
                         "WHERE workflow_id = %s ORDER BY firing_id", (wf,))


def decided(conn, monkeypatch, w, docs, ran, verb="approve", reason=None):
    """A clash paused for people and settled by them: approve = both members, reject = the first."""
    wf, out = call(monkeypatch, w, docs, ran)
    assert out["result"] == "paused_for_approval", out
    last = {w.key("A"): w.la2, w.key("B"): w.lb}
    ms = members_of(conn, wf)
    if verb == "approve":
        for m in ms:
            LG.act(conn, w, m["decision_id"], last[m["policy_name"]], reason=reason)
    else:
        LG.act(conn, w, ms[0]["decision_id"], last[ms[0]["policy_name"]], verb="reject", reason=reason or "No.")
    [lv] = live_of(conn, wf)
    return lv["decision_id"]


def _pair(w, **kw):
    return [LG.doc_a(w, **kw), LG.doc_b(w, **kw)]


def test_precedent_approves_after_n_person_approvals_and_the_tool_runs(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    first = decided(conn, monkeypatch, world, docs, ran)
    second = decided(conn, monkeypatch, world, docs, ran)
    assert ran == []
    wf, out = call(monkeypatch, world, docs, ran)
    assert ran == [LG.ARGS], "the tool runs now, with no person approving"
    assert out == {"refunded": 900}
    [lv] = live_of(conn, wf)
    assert (lv["status"], lv["decision"], lv["actioned_by"], lv["decision_scope"]) == \
           ("actioned", "approve", "system:precedent", "this_action")
    assert (lv["is_open"], lv["outcome"], lv["by_person"]) == (False, "approve", False)
    assert lv["facts"]["decidedBy"] == {"kind": "precedent", "name": "system:precedent"}
    cited = [e for e in lv["evidence"] if e.get("kind") == "precedent"]
    assert [e["caseId"] for e in cited] == [f"pc_{second}", f"pc_{first}"]
    assert {e["outcome"] for e in cited} == {"approve"} and all(e["actioned_by"].startswith("sub-") for e in cited)
    assert members_of(conn, wf) == []
    fs = firings_of(conn, wf)
    assert {f["result"] for f in fs} == {"allowed"} and {f["decision_id"] for f in fs} == {lv["decision_id"]}
    notes = LG.notes(conn, [f["firing_id"] for f in fs])
    assert {n["recipient"] for n in notes} == {OWNER, world.la1, world.la2, world.lb}
    assert all("ran on precedent" in n["message"] and "900" not in n["message"] for n in notes)


def test_the_agent_is_told_it_ran_on_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    wf = f"{world.wf}-direct"
    world.wfs.append(wf)
    LG.use(monkeypatch, docs)
    res = G.before_tool(tool_name=world.tool, args=dict(LG.ARGS), agent="overcharge_hunter", reason=LG.REASON,
                        workflow_id=wf, user_id="req@example.test")
    [lv] = live_of(conn, wf)
    assert res.allow is True
    assert res.to_agent == {"result": "allowed", "conflictCaseId": f"pc_{lv['decision_id']}", "precedent": True,
                            "precedentCount": 1}


def test_the_model_reads_the_precedent_note_after_the_tool_result(conn, world, monkeypatch, ran):
    """Final review I1 end to end: the tool message the model reads carries the result, then one note."""
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    wf = f"{world.wf}-note"
    world.wfs.append(wf)
    LG.use(monkeypatch, docs, world, ran)
    res, script = LG.run(monkeypatch, LG.stub_tools(world, ran), [LG._round(world.tool), LG.FINAL], workflow_id=wf)
    [lv] = live_of(conn, wf)
    assert res.calls[0].result == {"refunded": 900}
    [msg] = [m for m in script.seen[-1] if m.get("role") == "tool"]
    assert msg["content"] == ('{"refunded": 900}\nNote: this action ran without a person approving it, on precedent '
                              f"(decided the same way 2 times before, case pc_{lv['decision_id']}).")


def test_precedent_rejects_after_n_person_rejects(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    wf, out = call(monkeypatch, world, docs, ran)
    [lv] = live_of(conn, wf)
    assert ran == []
    assert out["result"] == "blocked" and out["reasonCode"] == "refused_on_precedent" and out["precedent"] is True
    assert out["conflictCaseId"] == f"pc_{lv['decision_id']}" and "refused on precedent" in out["reason"]
    assert (lv["decision"], lv["actioned_by"], lv["facts"]["decidedBy"]["kind"]) == ("reject", "system:precedent",
                                                                                      "precedent")
    fs = firings_of(conn, wf)
    assert {f["result"] for f in fs} == {"blocked"}
    assert all(f["reason"].startswith("refused_on_precedent") for f in fs)
    assert all("was refused on precedent" in n["message"] for n in LG.notes(conn, [f["firing_id"] for f in fs]))


def _escalated(conn, monkeypatch, w, docs, ran):
    wf, out = call(monkeypatch, w, docs, ran)
    assert out["result"] == "paused_for_approval" and ran == []
    [lv] = live_of(conn, wf)
    return lv


def test_below_n_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "only 2 of 3 decisions by people on this exact clash"}


def test_mixed_outcomes_go_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "decisions disagree"}
    assert len(lv["facts"]["history"]) == 2, "the people deciding see what came before"


def test_a_new_policy_version_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    newer = copy.deepcopy(docs)
    newer[0]["version"] = 2
    lv = _escalated(conn, monkeypatch, world, newer, ran)
    assert lv["facts"]["precedent"] == {"why": "only 0 of 2 decisions by people on this exact clash"}


def test_precedent_never_counts_toward_a_later_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    for _ in range(2):
        assert call(monkeypatch, world, docs, ran)[1] == {"refunded": 900}
    ran.clear()                                    # the two precedent runs; the next call must not run
    precedent_n(monkeypatch, 3)
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "only 2 of 3 decisions by people on this exact clash"}


@pytest.mark.parametrize("n,missing,why", [(0, False, "precedent is switched off"),
                                           (None, True, "precedent limit unavailable")])
def test_off_or_unreadable_goes_to_people(conn, world, monkeypatch, ran, n, missing, why):
    precedent_n(monkeypatch, n, missing=missing)
    lv = _escalated(conn, monkeypatch, world, _pair(world), ran)
    assert lv["facts"]["precedent"] == {"why": why}


def test_an_edit_takes_effect_without_a_restart(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    _escalated(conn, monkeypatch, world, docs, ran)
    precedent_n(monkeypatch, 2)                    # the row is edited; nothing restarts
    assert call(monkeypatch, world, docs, ran)[1] == {"refunded": 900}


def test_a_precedent_path_failure_refuses_and_leaves_nothing(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)

    def boom(*a, **k):
        raise RuntimeError("notification store down")
    monkeypatch.setattr(G, "_notify_precedent", boom)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out == G.UNAVAILABLE and ran == []
    assert live_of(conn, wf) == [] and members_of(conn, wf) == []
    assert [f["policy_key"] for f in firings_of(conn, wf)] == ["*"], "only the refusal is logged"


def test_a_failed_precedent_lookup_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    monkeypatch.setattr(DE, "PRECEDENT_SQL", "SELECT no_such_column FROM proc.bp_agent_policy_conflict "
                                             "WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %s")
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "precedent lookup failed"}, "the gate's transaction survived"


def test_a_repeat_call_with_open_member_cases_does_not_consult_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    wf, first = call(monkeypatch, world, docs, ran)
    assert first["result"] == "paused_for_approval"
    precedent_n(monkeypatch, 2)                    # precedent would now decide a NEW call
    _wf, again = call(monkeypatch, world, docs, ran, wf=wf)
    assert again == first and ran == []
    assert len(live_of(conn, wf)) == 1, "the repeat reuses the paused clash"


def test_the_replay_recheck_never_consults_precedent(conn, world, monkeypatch, ran):
    docs = _pair(world)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out["result"] == "paused_for_approval"
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("the replay consulted precedent"))
    replay = lambda d: R.run(d, agent_nick=object())   # noqa: E731
    last = {world.key("A"): world.la2, world.key("B"): world.lb}
    for m in members_of(conn, wf):
        LG.act(conn, world, m["decision_id"], last[m["policy_name"]], replay=replay)
    assert ran == [LG.ARGS]


def test_a_block_still_wins_and_precedent_is_not_consulted(conn, world, monkeypatch, ran):
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("precedent consulted on a block"))
    wf, out = call(monkeypatch, world, [LG.doc_a(world), LG.doc_b(world, outcome="block")], ran)
    assert out["result"] == "blocked" and ran == []
    [lv] = live_of(conn, wf)
    assert lv["facts"]["decidedBy"]["kind"] == "block"


def test_an_unreadable_block_beats_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    unreadable = LG.make_doc(world.key("U"), tool=world.tool, outcome="block", source=f"TST Legal {world.tag}")
    unreadable["trigger"]["condition"] = {"all": [{"field": "args.amount", "op": "between", "value": "x"}]}
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("precedent consulted on a block"))
    _wf, out = call(monkeypatch, world, docs + [unreadable], ran)
    assert out["result"] == "blocked" and ran == []


def test_precedent_never_approves_for_a_policy_outside_the_clash(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    # same source as A (tiered, no clash with A) and the same decider as B (no clash with B)
    third = LG.make_doc(world.key("T"), tool=world.tool, outcome="approve", deciders=[world.lb],
                        source=f"TST Finance {world.tag}")
    lv = _escalated(conn, monkeypatch, world, docs + [third], ran)
    assert lv["facts"]["precedent"] == {"why": f"{world.key('T')} also needs approval and is not part of this clash"}


# ---------------------------------------------------------------- carried from the reviews
def _always_approves(real):
    """decide_live_conflict as a precedent that approves whatever it is asked (calls the real one
    first, so a consult is visible as a real lookup), to prove a block wins even then."""
    asked = []

    def stub(cur, lc, *, ctx, now, requester):
        real(cur, lc, ctx=ctx, now=now, requester=requester)
        asked.append(lc.kind)
        return DE.Decision(subject_type=DE.LIVE_CONFLICT_SUBJECT, subject_id="x", decision="approve",
                           resolution=DE.RESOLVED, rationale="stub approve", facts={}, evidence=[])
    return stub, asked


def test_a_block_outside_the_clash_beats_a_precedent_approve(conn, world, monkeypatch, ran):
    """Carried (a): the clash A/B alone stays 'human' and precedent would approve it; a block
    that is NOT part of the clash is still in verdict.blocks, so the call is blocked and the
    engine is never asked."""
    from services.agent_policy import conflict_engine, enforcement
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    outside = LG.make_doc(world.key("X"), tool=world.tool, outcome="block", source=f"TST Finance {world.tag}")
    outside["trigger"]["condition"] = {"all": [{"field": "args.amount", "op": "between", "value": "x"}]}
    ctx = G._ctx(world.tool, dict(LG.ARGS), "overcharge_hunter", LG.REASON)
    verdict = enforcement.check(ctx, copy.deepcopy(docs + [outside]), default_response_time="PT4H")
    lc = conflict_engine.classify(verdict)
    assert lc.kind == "human" and [h["id"] for h in verdict.blocks] == [world.key("X")]
    assert world.key("X") not in {str(h["id"]) for h in lc.involved}, "the block is outside the clash"
    stub, asked = _always_approves(DE.decide_live_conflict)
    monkeypatch.setattr(DE, "decide_live_conflict", stub)
    wf, out = call(monkeypatch, world, docs + [outside], ran)
    assert out["result"] == "blocked" and out.get("reasonCode") != G.PRECEDENT_REFUSED and "precedent" not in out
    assert ran == [] and asked == [], "the tool never ran and precedent was never consulted"
    assert live_of(conn, wf) == [] and members_of(conn, wf) == []
    assert {f["result"] for f in firings_of(conn, wf)} == {"blocked"}


def test_a_block_from_one_approvers_source_is_recorded_and_still_blocks(conn, world, monkeypatch, ran):
    """Carried (a), the readable form: a block from A's own source does not clash with A, but it
    does with B, so the call is a block record (never 'human'), and precedent is never asked."""
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    same = LG.make_doc(world.key("Y"), tool=world.tool, outcome="block", source=f"TST Finance {world.tag}")
    stub, asked = _always_approves(DE.decide_live_conflict)
    monkeypatch.setattr(DE, "decide_live_conflict", stub)
    wf, out = call(monkeypatch, world, docs + [same], ran)
    assert out["result"] == "blocked" and ran == [] and asked == []
    [lv] = live_of(conn, wf)
    assert lv["facts"]["decidedBy"]["kind"] == "block"


def test_a_failing_precedent_engine_refuses_and_leaves_nothing(conn, world, monkeypatch, ran):
    """Carried (e): an exception from the engine itself (not its lookup) fails closed."""
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)

    def boom(*a, **k):
        raise RuntimeError("engine down")
    monkeypatch.setattr(DE, "decide_live_conflict", boom)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out == G.UNAVAILABLE and ran == []
    assert live_of(conn, wf) == [] and members_of(conn, wf) == []
    assert [f["policy_key"] for f in firings_of(conn, wf)] == ["*"], "only the refusal is logged"


def test_a_failing_precedent_record_refuses_and_leaves_nothing(conn, world, monkeypatch, ran):
    """Carried (e): the closed live record is written, then the firing link fails: one
    transaction, so the record is gone too."""
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    real = CL.insert_live
    made = []

    def spy(*a, **k):
        made.append(real(*a, **k))
        if k.get("actor") == CL.PRECEDENT:
            raise RuntimeError("after the precedent's live record")
        return made[-1]
    monkeypatch.setattr(CL, "insert_live", spy)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out == G.UNAVAILABLE and ran == [] and len(made) == 1
    assert LG.rows(conn, "SELECT decision_id FROM proc.bp_decision WHERE decision_id = %s", (made[0],)) == []
    assert live_of(conn, wf) == [] and members_of(conn, wf) == []


def test_the_escalated_record_stores_the_raw_history_with_its_private_fields(conn, world, monkeypatch, ran):
    """Carried (c): facts.history is conflict_history.raw (with _args), not a masked reading."""
    from services.agent_policy import conflict_detect, conflict_history
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran, reason="Paying the 900 refund is fine.")
    decided(conn, monkeypatch, world, docs, ran)
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    with conn.cursor() as cur:
        raw = conflict_history.raw(cur, pair_key=conflict_detect.pair_key(world.key("A"), world.key("B")),
                                   limit=conflict_history.IN_CASE_LIMIT)
    before = [e for e in raw if e["caseId"] != f"pc_{lv['decision_id']}"]
    stored = lv["facts"]["history"]
    assert [e["caseId"] for e in stored] == [e["caseId"] for e in before] and len(stored) == 2
    assert all(e["_args"] == {"amount": 900} for e in stored), "the raw entries keep _args"
    assert all("_owners" in e and "_deciders" in e for e in stored)
    assert stored[1]["decision"]["reason"] == "Paying the 900 refund is fine.", "unmasked as stored"


# ---------------------------------------------------------------- final review wave
@pytest.mark.parametrize("decider", ["lb", "la2"])   # lb: the credited (last) approver; la2: a member approver
def test_precedent_never_decides_for_a_requester_who_decided_a_cited_case(conn, world, monkeypatch, ran, decider):
    """C1: D approves 5 clashes others asked for; D's own request goes to people and the tool does
    not run. Anyone else's request still runs on precedent."""
    precedent_n(monkeypatch, 5)
    docs = _pair(world)
    for _ in range(5):
        decided(conn, monkeypatch, world, docs, ran)
    assert ran == []
    name = getattr(world, decider)
    me = LG.who(world, name).subject
    wf, out = call(monkeypatch, world, docs, ran, user_id=me)
    assert out["result"] == "paused_for_approval" and ran == [], "the tool must not run"
    [lv] = live_of(conn, wf)
    assert (lv["status"], lv["is_open"]) == ("open", True)
    assert lv["facts"]["precedent"] == {"why": "the requester decided an earlier case of this clash"}
    assert len(members_of(conn, wf)) == 2, "it went to people"
    assert len(lv["facts"]["history"]) == 5, "the people deciding see the five earlier cases"
    _wf, other = call(monkeypatch, world, docs, ran, user_id="someone-else@example.test")
    assert other == {"refunded": 900} and ran == [LG.ARGS], "precedent still decides for anyone else"


def _clocks(conn, live_id):
    [r] = LG.rows(conn, "SELECT d.created_at AS d_created, d.actioned_at, c.created_at AS c_created, c.decided_at "
                        "FROM proc.bp_decision d JOIN proc.bp_agent_policy_conflict c USING (decision_id) "
                        "WHERE d.decision_id = %s", (live_id,))
    return r


def test_a_closed_live_record_is_never_decided_before_it_was_raised(conn, world, monkeypatch, ran):
    """M1: precedent, block-record and standing-rule records write created_at and the decision
    time from one clock, so decidedAt is never earlier than raisedAt."""
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    wf_p, _ = call(monkeypatch, world, docs, ran)
    wf_b, _ = call(monkeypatch, world, [LG.doc_a(world), LG.doc_b(world, outcome="block")], ran)
    rule = {"with": world.key("B"), "prevails": world.key("A"),
            "rule": f"{world.key('A')} takes priority over {world.key('B')}"}
    wf_s, _ = call(monkeypatch, world, [LG.doc_a(world, conflicts=[rule]), LG.doc_b(world)], ran)
    for wf, kind in ((wf_p, "approve"), (wf_b, "block"), (wf_s, "standing_rule")):
        [lv] = live_of(conn, wf)
        assert lv["decision"] == kind
        c = _clocks(conn, lv["decision_id"])
        assert c["d_created"] == c["actioned_at"] == c["c_created"] == c["decided_at"], (kind, c)


def test_a_failed_history_read_never_refuses_the_call(conn, world, monkeypatch, ran, caplog):
    """M4: the escalate path's history snapshot fails (and poisons the transaction): rolled back to
    its savepoint, logged by type only, and the clash still goes to people, without history."""
    from services.agent_policy import conflict_history
    precedent_n(monkeypatch, 3)
    docs = _pair(world)

    def broken(cur, **kw):
        cur.execute("SELECT no_such_column_tst FROM proc.bp_agent_policy_conflict LIMIT 1")
    monkeypatch.setattr(conflict_history, "raw", broken)
    with caplog.at_level("ERROR"):
        wf, out = call(monkeypatch, world, docs, ran)
    assert out["result"] == "paused_for_approval" and ran == []
    [lv] = live_of(conn, wf)
    assert lv["facts"]["history"] == []
    assert lv["facts"]["precedent"] == {"why": "only 0 of 3 decisions by people on this exact clash"}
    assert len(members_of(conn, wf)) == 2
    assert "UndefinedColumn" in caplog.text and "no_such_column_tst" not in caplog.text


# ---------------------------------------------------------------- the governed value range (Task 12)
def call_for(monkeypatch, w, docs, ran, amount):
    """One scripted agent call for `amount`, in its own workflow."""
    wf = f"{w.wf}-{len(w.wfs)}"
    w.wfs.append(wf)
    LG.use(monkeypatch, docs, w, ran)
    res, _ = LG.run(monkeypatch, LG.stub_tools(w, ran),
                    [LG._round(w.tool, {**LG.ARGS, "amount": amount}), LG.FINAL], workflow_id=wf)
    return wf, res.calls[0].result


def test_precedent_applies_only_within_the_governed_value_range(conn, world, monkeypatch, ran):
    """Five clashes people approved at 900: 1000 (11% above) runs on precedent, 1200 (33% above)
    goes to people. The tool runs only for 1000."""
    precedent_n(monkeypatch, 5)
    precedent_range(monkeypatch, 20)
    docs = _pair(world)
    for _ in range(5):
        decided(conn, monkeypatch, world, docs, ran)
    assert ran == []

    wf, out = call_for(monkeypatch, world, docs, ran, 1000)
    assert out == {"refunded": 1000} and [r["amount"] for r in ran] == [1000]
    [lv] = live_of(conn, wf)
    assert (lv["decision"], lv["actioned_by"]) == ("approve", "system:precedent")
    assert lv["facts"]["valueRange"] == {"pct": 20.0, "fields": {
        "args.amount": {"value": 1000, "max": 900, "limit": 1080.0}}}
    from services.agent_policy import conflict_detect
    from services.agent_policy import conflict_history as CH
    with conn.cursor() as cur:
        entries = CH.read(cur, pair_key=conflict_detect.pair_key(world.key("A"), world.key("B")),
                          viewer=CH.ANONYMOUS)
    [e] = [x for x in entries if x["caseId"] == f"pc_{lv['decision_id']}"]
    sentence = "On precedent: decided the same way 5 times; within 20% of the largest earlier value"
    assert e["decision"]["reason"] == sentence
    assert e["valueRange"]["pct"] == 20.0 and set(e["valueRange"]["fields"]) == {"args.amount"}
    [row] = [r for r in CH.to_csv(entries).split("\r\n") if r.startswith(f'"pc_{lv["decision_id"]}"')]
    assert f'"{sentence}"' in row

    wf, out = call_for(monkeypatch, world, docs, ran, 1200)
    assert out["result"] == "paused_for_approval" and [r["amount"] for r in ran] == [1000]
    [lv] = live_of(conn, wf)
    assert (lv["status"], lv["is_open"]) == ("open", True)
    assert lv["facts"]["precedent"] == {"why": "args.amount 1200 is more than 20% above the largest earlier value (900)"}
    assert len(members_of(conn, wf)) == 2, "both approvals go to people"

    wf, out = call_for(monkeypatch, world, docs, ran, "1200")   # a number sent as text (review fix)
    assert out["result"] == "paused_for_approval" and [r["amount"] for r in ran] == [1000]
    [lv] = live_of(conn, wf)
    assert lv["facts"]["precedent"] == {"why": "args.amount 1200 is more than 20% above the largest earlier value (900)"}
