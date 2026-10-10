"""The suggestion queue, 'Draft this change', 'Not yet' and 'Dismiss' through the WHOLE app on bp_testdb.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. `draft` and the save hook read policy rows,
so each test has a REAL live policy: "TST …" in business area Operations, on a tool named
tst_<tag> (it can never apply to anyone's real action or contradict anyone's policy), RETIRED at
teardown. Suggestion rows of that policy are removed afterwards; firing rows are append-only BY
DESIGN and stay.

The numbers (the banding rule is worked in test_learning_scan_live's docstring): 33 approvals at
510, 520, ... 830 under "args.amount gt 500" -> band 1 = 510..800 -> the suggestion is 500 -> 800
on 30 decisions (30 approved) by 3 people over 40 days.
"""
import copy
import logging
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from repositories import agent_policy_repo as repo
from services.agent_policy import learning as L
from tests.agent_policy import fixtures as F
from tests.agent_policy.test_learning_scan_live import BASE, NOW, fire

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

# tst-buyer is linked to the test policy's owner in the decider map (the `world` fixture); tst-buyer-2
# is a Buyer linked to nobody. Stage 5b: only the owner or an Approver may hide a card.
ROLES = {"tst-viewer": "Viewer", "tst-buyer": "Buyer", "tst-buyer-2": "Buyer", "tst-approver": "Approver",
         "tst-admin": "Admin"}
EVIDENCE_30 = "30 of 30 requests between 500 USD and 800 USD were approved, by 3 different people over 40 days."


def _hdr(sub):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": f"{sub}@example.test", "X-User-Groups": "[]"}


VIEWER, BUYER, APPROVER = _hdr("tst-viewer"), _hdr("tst-buyer"), _hdr("tst-approver")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "learning endpoint tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def _form(tag):
    tool = f"tst_{tag}"
    form = copy.deepcopy(F.FORM_EXAMPLE)
    form["name"] = f"TST refund over 500 {tag}"
    form["businessArea"], form["subArea"] = "Operations", "General"
    form["owner"] = f"TST Owner {tag}"
    form["checked"] = {"by": "test", "at": "2026-10-09T08:00:00Z"}
    form["hidden"]["actions"]["tools"] = [tool]
    form["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": [tool]},
                                           {"field": "args.amount", "op": "gt", "value": 500}]}
    form["examples"] = [{"input": {"tool.name": tool, "args.amount": a}, "agentExpected": e, "flipped": False}
                        for a, e in ((501, "approve"), (500, "none"), (499, "none"))]
    return form


@pytest.fixture
def world(conn, monkeypatch):
    """A live approve policy "amount over 500" with the base evidence and its one open suggestion.
    Its owner name is linked to tst-buyer in the decider map, so BUYER acts as the policy's owner."""
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, audits=[], key=None)
    # tst_ tools are unknown to the real registry: readiness and the final check are bypassed
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: w.audits.append(kw))
    monkeypatch.setattr(R, "_role_of", lambda p: ROLES.get(p.subject, "Nobody"))
    form = _form(tag)
    try:
        w.key = repo.create_draft(conn, form, actor="test")["policyKey"]
        assert w.key.startswith("OPS-")
        repo.save_version(conn, w.key, form, base_version=1, intent="activate", actor="test", change_note="go")
        w.live = 2
        conn.cursor().execute(
            "INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) VALUES (%s, %s, 'test')",
            (form["owner"], ["tst-buyer@example.test"]))
        fire(conn, w.key, BASE, version=w.live)
        assert _scan(conn, w)["raised"] == 1
        [w.sid] = [c["id"] for c in L.for_policy(conn.cursor(), w.key)]
        yield w
    finally:
        # this test's decider-map rows: every name it links is "TST <word> <tag>"
        conn.cursor().execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name LIKE %s", (f"TST % {tag}",))
        if w.key:
            conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (w.key,))
            got = repo.get_policy(conn, w.key)
            if got["status"] != "retired":
                repo.retire(conn, w.key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


def _scan(conn, w):
    """The hourly scan, narrowed to this test's policy as the scan itself reads it (so `targets`,
    the compiled learning.eligible and the business area's setting are all exercised)."""
    [target] = [t for t in L.targets(conn.cursor()) if t.key == w.key]
    counts = L.scan(conn, NOW, only=[target])
    assert counts["errors"] == 0, counts
    return counts


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    return TestClient(app)


def _row(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT status, n_at_action, acted_by, saved_version FROM proc.bp_agent_policy_suggestion "
                "WHERE suggestion_id = %s", (w.sid,))
    return cur.fetchone()


def _policy(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT (SELECT count(*) FROM proc.bp_agent_policy_version WHERE policy_key = %s), status, "
                "live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s", (w.key, w.key))
    return cur.fetchone()


def _mine(client, w, headers=VIEWER, status=None):
    r = client.get("/agent-policies/suggestions" + (f"?status={status}" if status else ""), headers=headers)
    assert r.status_code == 200, r.text
    return [c for c in r.json()["suggestions"] if c["policyKey"] == w.key]


def _draft(client, w, headers=BUYER):
    return client.get(f"/agent-policies/suggestions/{w.sid}/draft", headers=headers)


def _act(client, w, verb, headers=BUYER):
    return client.post(f"/agent-policies/suggestions/{w.sid}/act", json={"verb": verb}, headers=headers)


def _save(client, w, form, *, base, headers=BUYER, intent="draft"):
    return client.post(f"/agent-policies/{w.key}/versions", headers=headers,
                       json={"form": form, "baseVersion": base, "intent": intent, "changeNote": "TST save"})


def _live_form(conn, w, **changes):
    form = copy.deepcopy(repo.get_policy(conn, w.key)["versions"][w.live - 1]["form"])
    if "limit" in changes:
        form["hidden"]["condition"]["all"][1]["value"] = changes.pop("limit")
    form.update(changes)
    return form


# ------------------------------------------------------------------ the queue and the roles
def test_viewer_reads_the_queue_buyer_acts_viewer_cannot(client, conn, world):
    [card] = _mine(client, world)
    assert card == {"id": world.sid, "policyKey": world.key, "policyName": f"TST refund over 500 {world.tag}",
                    "status": "open", "kind": "threshold",
                    "change": "Raise the limit on Refund amount from 500 USD to 800 USD?", "evidence": EVIDENCE_30,
                    "stats": {"decisions": 30, "approved": 30, "approvers": 3, "days": 40, "medianSeconds": 60.0,
                              "wilsonLower": 0.8865, "reversed": 0},
                    "suggestionConfidence": "High", "canDraft": True, "updatedAt": NOW.isoformat(),
                    # stage 5b, as the Viewer reading it: not theirs, and not theirs to hide
                    "ownerName": f"TST Owner {world.tag}", "mine": False, "canAct": False}
    assert client.get("/agent-policies/suggestions", headers=_hdr("tst-nobody")).status_code == 403
    assert _act(client, world, "not_yet", VIEWER).status_code == 403
    assert _draft(client, world, VIEWER).status_code == 403
    assert _row(conn, world) == ("open", None, None, None), "a refused caller changes nothing"
    r = _act(client, world, "not_yet")
    assert r.status_code == 201, r.text
    assert (r.json()["id"], r.json()["status"], r.json()["canDraft"]) == (world.sid, "not_yet", False)
    assert _row(conn, world) == ("not_yet", 30, "tst-buyer", None)
    assert _mine(client, world) == [] and [c["status"] for c in _mine(client, world, status="all")] == ["not_yet"]


def test_acting_is_audited_before_and_after_and_reading_is_not(client, world):
    _mine(client, world)
    _draft(client, world)
    assert world.audits == [], "reads are not audit events"
    _act(client, world, "dismiss", VIEWER)
    assert [(a["action_type"], a["phase"], a["status"]) for a in world.audits] == [
        ("agent_policy.suggestion_act", "authorize", "denied")]
    del world.audits[:]
    assert _act(client, world, "dismiss").status_code == 201
    assert [(a["action_type"], a["phase"], a["status"]) for a in world.audits] == [
        ("agent_policy.suggestion_act", "authorize", "allowed"), ("agent_policy.suggestion_act", "act", "done")]
    assert world.audits[0]["details"]["suggestion"] == world.sid and world.audits[0]["details"]["verb"] == "dismiss"
    assert world.audits[1]["details"]["policy"] == world.key
    del world.audits[:]
    r = _act(client, world, "not_yet")
    assert r.status_code == 409 and r.json() == {"code": "not_open", "message": "This suggestion is no longer open."}
    assert [(a["phase"], a["status"], a["details"].get("code")) for a in world.audits] == [
        ("authorize", "allowed", None), ("act", "refused", "not_open")]


def test_the_policy_page_carries_its_open_cards(client, world):
    [card] = _mine(client, world)
    got = client.get(f"/agent-policies/{world.key}", headers=VIEWER)
    assert got.status_code == 200, got.text
    assert got.json()["suggestions"] == [card] and got.json()["suggestions"][0]["id"] == world.sid
    assert _act(client, world, "dismiss").status_code == 201
    assert client.get(f"/agent-policies/{world.key}", headers=VIEWER).json()["suggestions"] == []


def test_a_weak_card_says_not_enough_yet_with_todays_evidence_and_refuses_the_draft(client, conn, world):
    # A rejection at 605 lands in band 1 (now 510..600, 605, 610..790): 29 of 30 approved, Wilson
    # lower 0.8333 < 0.85, so nothing is proposed. The stored change's released set (500, 800] is
    # 31 decisions with 30 approved. Wilson lower, p = 30/31 = 0.967742, z^2 = 3.8415:
    # (p + z^2/62 - 1.96 * sqrt(p(1-p)/31 + z^2/3844)) / (1 + z^2/31)
    # = (0.967742 + 0.061959 - 1.96 * 0.044792) / 1.123919 = 0.941909 / 1.123919 = 0.8381.
    fire(conn, world.key, [605], result="rejected", first=33, version=world.live)
    assert _scan(conn, world)["weakened"] == 1
    [card] = _mine(client, world)
    assert (card["suggestionConfidence"], card["canDraft"]) == ("Not enough yet", False)
    assert card["evidence"] == ("30 of 31 requests between 500 USD and 800 USD were approved, "
                                "by 3 different people over 40 days.")
    assert (card["stats"]["decisions"], card["stats"]["approved"], card["stats"]["wilsonLower"]) == (31, 30, 0.8381)
    r = _draft(client, world)
    assert r.status_code == 409 and r.json() == {"code": "not_confident", "message": "There is not enough evidence yet."}


# ------------------------------------------------------------------ Draft this change
def test_draft_returns_a_form_and_writes_nothing(client, conn, world):
    before = _policy(conn, world)
    r = _draft(client, world)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"policyKey", "baseVersion", "form", "changeNote", "checkWording", "changedWording",
                      "suggestionId"}
    assert (d["policyKey"], d["baseVersion"], d["suggestionId"]) == (world.key, world.live, world.sid)
    assert d["form"]["hidden"]["condition"]["all"][1] == {"field": "args.amount", "op": "gt", "value": 800}
    assert d["form"]["checked"] is None, "the owner must read it again"
    # "above $500" appears once in the situation; each message says "over $500" once
    assert d["form"]["situation"] == "The agent is about to issue a refund or credit above $800."
    assert d["changedWording"] == ["situation", "messageForAgent"] and d["checkWording"] == ["messageForPerson"]
    assert d["changeNote"] == ("Suggested by learning: Raise the limit on Refund amount from 500 USD to 800 USD? "
                               f"Evidence: {EVIDENCE_30}")
    amounts = [e["input"]["args.amount"] for e in d["form"]["examples"]]
    assert amounts[-3:] == [799, 800, 801] and [e["agentExpected"] for e in d["form"]["examples"]][-3:] == \
        ["none", "none", "approve"]
    assert _policy(conn, world) == before, "no version written, the pointers unmoved"
    assert _row(conn, world) == ("open", None, None, None)
    assert client.get(f"/agent-policies/{world.key}", headers=VIEWER).json()["latestVersion"] == world.live


def test_saving_the_draft_marks_the_suggestion_saved_and_it_is_not_suggested_again(client, conn, world):
    d = _draft(client, world).json()
    r = _save(client, world, d["form"], base=d["baseVersion"])
    assert r.status_code in (200, 201), r.text
    assert r.json()["version"] == world.live + 1
    assert _row(conn, world) == ("saved", None, None, world.live + 1)
    got = client.get(f"/agent-policies/{world.key}", headers=VIEWER).json()
    assert got["suggestions"] == [] and got["liveVersion"] == world.live, "saved as a draft, not activated"
    assert _mine(client, world) == [] and [c["status"] for c in _mine(client, world, status="all")] == ["saved"]
    # 40 more approvals (840..1230): bands 510..800 and 810..1100 both pass, which would raise or
    # reopen any card that was not saved (the live limit is still 500: the draft is not live)
    fire(conn, world.key, [840 + 10 * i for i in range(40)], first=33, version=world.live)
    counts = _scan(conn, world)
    assert (counts["raised"], counts["updated"], counts["reopened"]) == (0, 0, 0)
    assert _row(conn, world) == ("saved", None, None, world.live + 1)
    r = _draft(client, world)
    assert r.status_code == 409 and r.json()["code"] == "policy_changed"


def test_saving_an_unrelated_edit_does_not_mark_it_saved(client, conn, world):
    r = _save(client, world, _live_form(conn, world, owner=f"TST Other Owner {world.tag}"), base=world.live)
    assert r.status_code in (200, 201), r.text
    assert _row(conn, world) == ("open", None, None, None)
    # the newer draft still holds "over 500", so the change is drafted on top of it
    d = _draft(client, world)
    assert d.status_code == 200 and d.json()["baseVersion"] == world.live + 1
    assert d.json()["form"]["owner"] == f"TST Other Owner {world.tag}"
    # stage 5b: a draft that still starts where the change starts is not in the way
    [card] = _mine(client, world)
    assert card["canDraft"] is True and "blocked" not in card


def test_a_policy_that_moved_on_refuses_the_draft(client, conn, world):
    # the owner raises the limit by hand to 700: looser than 500 in the suggested direction, so that
    # save holds the change and the suggestion is closed as saved; nothing stale is drafted
    assert _save(client, world, _live_form(conn, world, limit=700), base=world.live).status_code in (200, 201)
    assert _row(conn, world)[0] == "saved"
    r = _draft(client, world)
    assert r.status_code == 409
    assert r.json() == {"code": "policy_changed", "message": "This policy has changed since the suggestion was made."}


def test_a_policy_tightened_by_hand_refuses_the_draft_and_the_card_stays_until_the_scan(client, conn, world):
    # 400 is the other way: not the suggested change, so not "saved"; but the form no longer says 500
    assert _save(client, world, _live_form(conn, world, limit=400), base=world.live).status_code in (200, 201)
    assert _row(conn, world)[0] == "open"
    r = _draft(client, world)
    assert r.status_code == 409 and r.json()["code"] == "policy_changed"
    # stage 5b: the card says so before anyone clicks, in the queue and on the policy's page
    [card] = _mine(client, world)
    [paged] = client.get(f"/agent-policies/{world.key}", headers=VIEWER).json()["suggestions"]
    for c in (card, paged):
        assert (c["canDraft"], c["blocked"]) == (False, "The latest draft of this policy already changes this. "
                                                        "Activate or discard that draft first.")


def test_a_retired_policy_refuses_the_draft(client, conn, world):
    [before] = _mine(client, world)
    assert before["canDraft"] is True and "blocked" not in before
    repo.retire(conn, world.key, base_version=world.live, actor="test", change_note="test")
    r = _draft(client, world)
    assert r.status_code == 409 and r.json()["code"] == "policy_changed"
    [card] = _mine(client, world)                                       # stage 5b
    assert (card["canDraft"], card["blocked"]) == (False, "This policy is no longer Active.")


def test_nothing_in_this_stage_activates_a_policy(client, conn, world):
    before = _policy(conn, world)
    assert before[1:] == ("live", world.live, world.live)
    fire(conn, world.key, [840 + 10 * i for i in range(40)], first=33, version=world.live)
    _scan(conn, world)
    assert _draft(client, world).status_code == 200
    assert _act(client, world, "not_yet").status_code == 201
    assert _policy(conn, world) == before, "no version row, live_version unmoved"


def test_a_live_policy_that_moved_off_the_suggested_limit_refuses_the_draft_even_if_a_draft_went_back(client, conn, world):
    # activated at 400 (tighter: not the suggested change, so the card is not "saved"), then a draft
    # back at 500. The latest form says 500 again, but the policy in force does not: nothing is drafted.
    r = _save(client, world, _live_form(conn, world, limit=400), base=world.live, headers=APPROVER, intent="activate")
    assert r.status_code in (200, 201), r.text
    assert _save(client, world, _live_form(conn, world), base=world.live + 1).status_code in (200, 201)
    assert _policy(conn, world)[1:] == ("live", world.live + 1, world.live + 2) and _row(conn, world)[0] == "open"
    r = _draft(client, world)
    assert r.status_code == 409 and r.json()["code"] == "policy_changed"


def test_a_latest_version_that_no_longer_asks_for_approval_refuses_the_draft(client, conn, world):
    form = _live_form(conn, world, outcome="block", deciders=[])        # the limit is still 500
    assert _save(client, world, form, base=world.live).status_code in (200, 201)
    assert _row(conn, world)[0] == "open"
    r = _draft(client, world)
    assert r.status_code == 409 and r.json()["code"] == "policy_changed"


# ------------------------------------------------------------------ saved means in a version in force
def test_a_draft_saved_and_then_saved_back_brings_the_card_back(client, conn, world):
    d = _draft(client, world).json()
    assert _save(client, world, d["form"], base=d["baseVersion"]).status_code in (200, 201)      # v3 at 800
    assert _row(conn, world) == ("saved", None, None, world.live + 1)
    counts = _scan(conn, world)
    assert counts["reopened"] == 0 and _row(conn, world)[0] == "saved", "the latest draft holds the change"
    # the person saves it back at 500 and activates that: no version in force holds the change
    r = _save(client, world, _live_form(conn, world), base=world.live + 1, headers=APPROVER, intent="activate")
    assert r.status_code in (200, 201), r.text
    assert _policy(conn, world)[1:] == ("live", world.live + 2, world.live + 2)
    assert _row(conn, world) == ("saved", None, None, world.live + 1), "a save alone reopens nothing"
    # v2 and v4 are live versions with the identical condition, so v2's 33 decisions are the evidence
    counts = _scan(conn, world)
    assert (counts["reopened"], counts["raised"]) == (1, 0)
    assert _row(conn, world) == ("open", None, None, None)
    [card] = _mine(client, world)
    assert (card["id"], card["canDraft"], card["evidence"]) == (world.sid, True, EVIDENCE_30)
    assert _draft(client, world).json()["baseVersion"] == world.live + 2


def test_a_machine_draft_that_is_replaced_does_not_silence_the_card(conn, world):
    # the path a re-extraction takes: repo.save_version, intent draft, twice. The first holds a
    # looser limit; the second "replaces the edits" back to 500. Nobody dismissed anything.
    looser, back = _live_form(conn, world, limit=800), _live_form(conn, world)
    repo.save_version(conn, world.key, looser, base_version=world.live, intent="draft", actor="tst-extraction",
                      change_note="Re-extracted")
    assert _row(conn, world) == ("saved", None, None, world.live + 1)
    assert _scan(conn, world)["reopened"] == 0
    repo.save_version(conn, world.key, back, base_version=world.live + 1, intent="draft", actor="tst-extraction",
                      change_note="Re-extracted. Replaces the edits in v3.")
    assert _row(conn, world)[0] == "saved"
    assert _scan(conn, world)["reopened"] == 1
    assert _row(conn, world) == ("open", None, None, None)
    assert [c["id"] for c in L.for_policy(conn.cursor(), world.key)] == [world.sid]


# ------------------------------------------------------------------ the save hook
def test_a_save_never_fails_because_of_learning_bookkeeping(client, conn, world, monkeypatch, caplog):
    def _boom(cur, key, version, form):
        cur.execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = %s "
                    "WHERE policy_key = %s", (version, key))
        cur.execute("SELECT 1 / 0")            # fails inside the save's own transaction
    monkeypatch.setattr(L, "note_save", _boom)
    d = _draft(client, world).json()
    with caplog.at_level(logging.ERROR):
        r = _save(client, world, d["form"], base=d["baseVersion"])
    assert r.status_code in (200, 201), r.text
    assert _policy(conn, world)[3] == world.live + 1, "the person's save went through"
    assert _row(conn, world) == ("open", None, None, None), "the half-done bookkeeping was undone, not half-kept"
    assert [rec for rec in caplog.records if world.key in rec.getMessage() and rec.exc_info], "and it was logged"


def test_a_refused_save_marks_nothing(client, conn, world):
    d = _draft(client, world).json()
    assert _save(client, world, d["form"], base=d["baseVersion"] + 5).status_code == 409      # stale base
    assert _row(conn, world) == ("open", None, None, None)


# ------------------------------------------------------------------ shapes
def test_unknown_suggestion_is_404_and_a_bad_verb_is_422(client, world):
    gone = 10 ** 17
    assert client.get(f"/agent-policies/suggestions/{gone}/draft", headers=BUYER).status_code == 404
    r = client.post(f"/agent-policies/suggestions/{gone}/act", json={"verb": "dismiss"}, headers=BUYER)
    assert r.status_code == 404 and r.json() == {"detail": "No such suggestion."}
    assert _act(client, world, "apply").status_code == 422
    assert client.post(f"/agent-policies/suggestions/{world.sid}/act", json={}, headers=BUYER).status_code == 422
    assert client.get("/agent-policies/suggestions?status=closed", headers=VIEWER).status_code == 422
    assert client.get("/agent-policies/suggestions/abc/draft", headers=BUYER).status_code == 422
    assert client.get(f"/agent-policies/suggestions/{10 ** 30}/draft", headers=BUYER).status_code == 422


# ------------------------------------------------------------------ the scan judges under its lock
def _stale_target(conn, w):
    """The policy as a scan read it a moment ago, before whatever the test does next."""
    [target] = [t for t in L.targets(conn.cursor()) if t.key == w.key]
    return target


def test_a_save_landing_after_the_scan_read_its_policies_does_not_reopen_the_saved_card(client, conn, world):
    stale = _stale_target(conn, world)                       # live 500, nothing newer
    d = _draft(client, world).json()
    assert _save(client, world, d["form"], base=d["baseVersion"]).status_code in (200, 201)      # v3 at 800
    assert _row(conn, world) == ("saved", None, None, world.live + 1)
    # The scan still holds the policy as it read it. The same 33 decisions still support 500 -> 800,
    # so judged on the stale picture (latest = 500) the saved card would be reopened.
    counts = L.scan(conn, NOW, only=[stale])
    assert (counts["reopened"], counts["updated"], counts["errors"]) == (0, 0, 0)
    assert _row(conn, world) == ("saved", None, None, world.live + 1), "the latest version, read under the lock, holds it"


def test_a_policy_whose_live_condition_changed_since_the_scan_read_it_is_left_for_the_next_scan(client, conn, world,
                                                                                               caplog):
    stale = _stale_target(conn, world)                       # live 500
    r = _save(client, world, _live_form(conn, world, limit=700), base=world.live, headers=APPROVER, intent="activate")
    assert r.status_code in (200, 201), r.text
    before = _row(conn, world)
    assert before[0] == "saved", "700 is looser than 500: the save holds the change"
    with caplog.at_level(logging.INFO, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[stale])
    # evidence read for "over 500" says nothing about a policy that now says "over 700"
    assert counts == {**{k: 0 for k in L.COUNTERS}, "policies": 1, "skipped": 1}
    assert _row(conn, world) == before
    assert [r for r in caplog.records if world.key in r.getMessage() and "next scan" in r.getMessage()]
    # the next scan reads the policy afresh: the 500 card is closed for good or kept saved, never reopened at 500
    assert _scan(conn, world)["reopened"] == 0 and _row(conn, world)[0] == "saved"


def test_a_policy_whose_live_version_changed_since_the_scan_read_it_is_left_for_the_next_scan(client, conn, world):
    stale = _stale_target(conn, world)                       # live version 2, limit 500
    # the same condition activated again as a new version: the versions whose firings are the
    # evidence were listed for the old one
    r = _save(client, world, _live_form(conn, world, owner=f"TST Other owner {world.tag}"), base=world.live,
              headers=APPROVER, intent="activate")
    assert r.status_code in (200, 201), r.text
    assert _policy(conn, world)[2] == world.live + 1 and _row(conn, world)[0] == "open"
    counts = L.scan(conn, NOW, only=[stale])
    assert counts == {**{k: 0 for k in L.COUNTERS}, "policies": 1, "skipped": 1}
    fresh = _scan(conn, world)                               # the next scan: both versions' firings count
    assert (fresh["skipped"], fresh["updated"]) == (0, 1) and _row(conn, world)[0] == "open"
