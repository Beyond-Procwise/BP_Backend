"""Policy conflict cases against bp_testdb: raised on save, deduplicated per pair, routed to owners.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Policies are made through repo.create_draft
(ids can never be deleted, so they are RETIRED at teardown) with tools named tst_<tag>, so they
can never contradict anyone else's policy, and every detection here passes among= so it never
looks at anyone else's. Cases, conflict rows, notifications and decider-map rows made here are
removed afterwards.
"""
import copy
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from repositories import agent_policy_repo as repo
from services import policy_condition as pc
from services.agent_policy import approvals, conditions, conflict_cases as CC, conflict_payload
from tests.agent_policy.fixtures import FORM_EXAMPLE

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "conflict live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn):
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, tool=f"tst_{tag}", owner_a=f"TST Owner A {tag}", owner_b=f"TST Owner B {tag}",
                        keys=[])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test'), (%s, '{}', %s, 'test')",
                    (w.owner_a, [f"a-{tag}@example.test"], w.owner_b, [f"b-{tag}@example.test"]))
    yield w
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id FROM proc.bp_agent_policy_conflict WHERE policy_keys && %s", (w.keys,))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id = ANY(%s)",
                    (CC.SUBJECT_POLICY, [s for s in _pair_keys(w.keys)]))
        ids = sorted(set(ids) | {r[0] for r in cur.fetchall()})
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE firing_id IS NULL AND link = ANY(%s)",
                    ([f"conflict:{i}" for i in ids],))
        cur.execute("DELETE FROM proc.bp_agent_policy_conflict WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(%s)", ([w.owner_a, w.owner_b],))
    for key in w.keys:
        got = repo.get_policy(conn, key)
        if got["status"] != "retired":
            repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


def _pair_keys(keys):
    return {f"{a}|{b}" for a in keys for b in keys if a < b}


def _form(w, *, outcome, gt, doc, owner, reference="1.1", excerpt=None, amount_op="gt"):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["name"] = f"TST {outcome} over {gt} {w.tag}"
    form["businessArea"], form["subArea"] = None, None          # Unassigned (GEN-): no real area's numbers used
    form["outcome"] = outcome
    form["deciders"] = ["Finance Manager", "CFO"] if outcome == "approve" else []
    form["notify"] = []
    form["owner"] = owner
    form["source"] = {"document": doc, "documentVersion": 1, "reference": reference,
                      "excerpt": excerpt or f"Refunds  above ${gt} \"need\" a decision ({outcome})."}
    form["hidden"]["actions"]["tools"] = [w.tool]
    form["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": [w.tool]},
                                           {"field": "args.amount", "op": amount_op, "value": gt}]}
    form["examples"] = [{"input": {"tool.name": w.tool, "args.amount": gt + 1}, "agentExpected": outcome,
                         "flipped": False}]
    return form


def _make(conn, w, **kw):
    key = repo.create_draft(conn, _form(w, **kw), actor="test")["policyKey"]
    w.keys.append(key)
    return key


def _fin_cus(conn, w, *, owner_b=None):
    a = _make(conn, w, outcome="approve", gt=500, doc=f"TST Finance {w.tag}", owner=w.owner_a)
    b = _make(conn, w, outcome="block", gt=10000, doc=f"TST Customer {w.tag}", owner=owner_b or w.owner_b)
    return a, b


def _cases(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id, subject_id, status, decision, resolution, rationale, policy_name, facts, "
                    "evidence, options, respond_by, on_timeout, decision_scope, created_by, agent "
                    "FROM proc.bp_decision WHERE subject_type = %s AND subject_id = ANY(%s) ORDER BY decision_id",
                    (CC.SUBJECT_POLICY, list(_pair_keys(w.keys))))
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def _conflict_rows(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM proc.bp_agent_policy_conflict WHERE policy_keys && %s ORDER BY decision_id",
                    (w.keys,))
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def _notes(conn, did):
    with conn.cursor() as cur:
        cur.execute("SELECT recipient, message, firing_id FROM proc.bp_policy_notification WHERE link = %s "
                    "ORDER BY notification_id", (f"conflict:{did}",))
        return cur.fetchall()


def _doc(conn, key):
    got = repo.get_policy(conn, key)
    return got["versions"][-1]["compiled"]


def test_save_raises_policy_case_with_code_found_witness(conn, world):
    a, b = _fin_cus(conn, world)
    ids = CC.detect_for(conn, b, now=NOW, among=[a])
    assert len(ids) == 1
    cases = _cases(conn, world)
    assert len(cases) == 1
    c = cases[0]
    key = "|".join(sorted([a, b]))
    assert c["decision_id"] == ids[0] and c["status"] == "open" and c["subject_id"] == key
    assert c["decision"] == "resolve_conflict" and c["resolution"] == "escalated" and c["policy_name"] == key
    assert c["created_by"] == "system:conflict_detector" and c["agent"] == "agent_policy_conflicts"
    assert c["respond_by"] is None and c["on_timeout"] == "none" and c["decision_scope"] is None
    da, db = _doc(conn, a), _doc(conn, b)
    assert c["rationale"] == conflict_payload.why_line([da, db])
    # the witness is a real input both policies match, re-evaluated with the one evaluator
    ex = c["evidence"][0]["example"]
    assert c["evidence"][0]["kind"] == "overlap"
    for d in (da, db):
        assert pc.evaluate(conditions.to_engine(d["trigger"]["condition"]), conditions.nest(ex)) is True
    pols = {p["id"]: p for p in c["facts"]["policies"]}
    assert set(pols) == {a, b}
    assert pols[a]["source"]["excerpt"] == da["source"]["excerpt"]          # verbatim, odd spacing and quotes
    assert pols[b]["source"]["excerpt"] == db["source"]["excerpt"]
    assert c["facts"]["schema"] == "policy-conflict/1" and "unroutable" not in c["facts"]
    first, second = sorted([a, b])
    assert c["options"] == conflict_payload.policy_options(*(da if k == a else db for k in (first, second)))
    assert c["options"][0] == f"keep_both:{b}" and len(c["options"]) == 7     # Q1: only the block's keep_both
    rows = _conflict_rows(conn, world)
    assert len(rows) == 1 and rows[0]["kind"] == "policy" and rows[0]["raised_by"] == "save"
    assert rows[0]["is_open"] is True and rows[0]["pair_key"] == key
    assert sorted(rows[0]["policy_keys"]) == [first, second] and rows[0]["policy_versions"] == {a: 1, b: 1}
    notes = _notes(conn, ids[0])
    assert sorted(n[0] for n in notes) == sorted([world.owner_a, world.owner_b])
    assert all(n[2] is None and n[1] == f"Policies {first} and {second} conflict; a decision is needed."
               for n in notes)


def test_tiered_same_source_raises_nothing(conn, world):
    doc = f"TST Finance {world.tag}"
    a = _make(conn, world, outcome="approve", gt=500, doc=doc, owner=world.owner_a, reference="1.1")
    b = _make(conn, world, outcome="block", gt=10000, doc=doc.upper() + "  ", owner=world.owner_b, reference="1.2")
    assert CC.detect_for(conn, b, now=NOW, among=[a]) == []
    assert _cases(conn, world) == [] and _conflict_rows(conn, world) == []


def test_same_outcome_raises_nothing(conn, world):
    a = _make(conn, world, outcome="approve", gt=500, doc=f"TST Finance {world.tag}", owner=world.owner_a)
    b = _make(conn, world, outcome="approve", gt=10000, doc=f"TST Customer {world.tag}", owner=world.owner_b)
    assert CC.detect_for(conn, b, now=NOW, among=[a]) == []
    assert _cases(conn, world) == []


def test_resave_does_not_duplicate_open_case(conn, world):
    a, b = _fin_cus(conn, world)
    assert len(CC.detect_for(conn, b, now=NOW, among=[a])) == 1
    repo.save_version(conn, b, _form(world, outcome="block", gt=9000, doc=f"TST Customer {world.tag}",
                                     owner=world.owner_b), base_version=1, intent="draft", actor="test",
                      change_note="resave")
    assert CC.detect_for(conn, b, now=NOW, among=[a]) == []
    assert CC.detect_for(conn, a, now=NOW, among=[b], raised_by="scan") == []
    assert len(_cases(conn, world)) == 1 and len(_conflict_rows(conn, world)) == 1


def test_concurrent_raise_one_case_per_pair(conn, world, monkeypatch):
    from services.db import get_conn
    a, b = _fin_cus(conn, world)
    # widen the window between the lock and the checks, so two unserialised raises would both pass them
    monkeypatch.setattr(CC, "_after_lock", lambda key: time.sleep(0.3))
    barrier = threading.Barrier(2)
    results, errors = [], []

    def run(saved, other):
        try:
            with get_conn() as c:
                barrier.wait()
                results.append(CC.detect_for(c, saved, now=NOW, among=[other]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(b, a)), threading.Thread(target=run, args=(a, b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert sorted(len(r) for r in results) == [0, 1]
    assert len([c for c in _cases(conn, world) if c["status"] == "open"]) == 1
    assert len(_conflict_rows(conn, world)) == 1


def _decide_by_sql(conn, did, outcome="keep_both"):
    """Stand-in for Task 5's decide: close the case and its conflict row as a person would."""
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_decision SET status = 'actioned' WHERE decision_id = %s", (did,))
        cur.execute("UPDATE proc.bp_agent_policy_conflict SET is_open = false, outcome = %s, decided_by = 'test', "
                    "decided_at = now(), by_person = true WHERE decision_id = %s", (outcome, did))


def test_decided_pair_re_raised_only_after_a_new_version_still_overlaps(conn, world):
    a, b = _fin_cus(conn, world)
    [first] = CC.detect_for(conn, b, now=NOW, among=[a])
    _decide_by_sql(conn, first, f"change:{a}")
    # nothing changed since the decision: no new case, from either side
    assert CC.detect_for(conn, b, now=NOW, among=[a]) == []
    assert CC.detect_for(conn, a, now=NOW, among=[b], raised_by="scan") == []
    # a new version that no longer overlaps raises nothing
    repo.save_version(conn, a, _form(world, outcome="approve", gt=10000, doc=f"TST Finance {world.tag}",
                                     owner=world.owner_a, amount_op="lte"),
                      base_version=1, intent="draft", actor="test", change_note="narrowed")
    assert CC.detect_for(conn, a, now=NOW, among=[b]) == []
    # a newer version that still overlaps raises a new case, recording the new versions
    repo.save_version(conn, a, _form(world, outcome="approve", gt=600, doc=f"TST Finance {world.tag}",
                                     owner=world.owner_a), base_version=2, intent="draft", actor="test",
                      change_note="widened again")
    [second] = CC.detect_for(conn, a, now=NOW, among=[b])
    assert second != first
    rows = {r["decision_id"]: r for r in _conflict_rows(conn, world)}
    assert rows[second]["is_open"] and rows[second]["policy_versions"] == {a: 3, b: 1}


def test_owner_unlinked_marks_unroutable_and_notifies_administrators(conn, world):
    nobody = f"TST Unlinked Owner {world.tag}"
    a, b = _fin_cus(conn, world, owner_b=nobody)
    [did] = CC.detect_for(conn, b, now=NOW, among=[a])
    [c] = _cases(conn, world)
    assert c["facts"]["unroutable"] == [nobody]
    notes = _notes(conn, did)
    recipients = [n[0] for n in notes]
    assert sorted(recipients) == sorted([world.owner_a, nobody, approvals.ADMIN_RECIPIENT])
    admin = next(n for n in notes if n[0] == approvals.ADMIN_RECIPIENT)
    assert nobody in admin[1] and all(n[2] is None for n in notes)
    assert "10000" not in admin[1] and "10001" not in admin[1]      # no input values in notification text


def test_detect_all_scans_and_never_raises(conn, world, monkeypatch):
    a, b = _fin_cus(conn, world)
    # the scan normally covers everything; here it is narrowed to this test's policies
    real = CC._detect
    monkeypatch.setattr(CC, "_detect", lambda c, key, **kw: real(c, key, **{**kw, "among": [a, b]})
                        if key in (a, b) else [])
    stats = CC.detect_all(conn, now=NOW)
    assert stats["raised"] == 1 and stats["errors"] == 0 and stats["pairs"] >= 2
    rows = _conflict_rows(conn, world)
    assert len(rows) == 1 and rows[0]["raised_by"] == "scan"
