"""The second reviewer on bp_testdb, through the real save path. Needs PROCWISE_TEST_LIVE_DB=1.

Each test has REAL live policies ("TST …", on a tool named tst_<tag>, RETIRED at teardown): one in
Finance (second reviewer on, by seed) and, where it says so, one in Operations (off). Each gets
the standard evidence and its one suggestion from the real scan (worked in
test_learning_scan_live's docstring): 33 approvals at 510..830 under "args.amount gt 500" ->
the suggestion is 500 -> 800. So 800 is the suggested loosening, 600 a smaller one, 2000 goes
beyond it, 900 is "a different condition from the one reviewed", 400 is a tightening.

Review rows, suggestion rows and the decider-map row a test makes are removed by the fixture;
firing rows are append-only BY DESIGN and stay.
"""
import copy
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.auth import Principal
from api.routers import agent_policies as R
from repositories import agent_policy_repo as repo
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from tests.agent_policy.test_learning_endpoints_live import _form
from tests.agent_policy.test_learning_scan_live import BASE, NOW, fire

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REQUIRED = [{"field": "secondReview", "code": "second_review_required", "message": SR.MESSAGE}]
UNAVAILABLE = [{"field": "secondReview", "code": "second_review_unavailable",
                "message": "Can't check whether a second review is needed right now. Try again shortly."}]
NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}
ROLES = {"tst-buyer": "Buyer", "tst-saver": "Approver", "tst-reviewer": "Approver"}


def person(sub, email=None):
    return Principal(subject=sub, email=email or f"{sub}@example.test", claims={"cognito:groups": []})


SAVER, REVIEWER = person("tst-saver"), person("tst-reviewer")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "second-review tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def make(conn, monkeypatch):
    """make(area) -> a live approve policy "amount over 500" in that business area, with the base
    evidence, its one open suggestion, and a person linked to its owner."""
    cur = conn.cursor()
    cur.execute("SELECT area_name, second_reviewer FROM proc.bp_business_area WHERE area_name IN ('Finance','Operations')")
    assert dict(cur.fetchall()) == {"Finance": True, "Operations": False}, \
        "these tests need the seeded settings: second reviewer on for Finance, off for Operations"
    # tst_ tools are unknown to the real registry: readiness and the final check are bypassed.
    # The second-review check is NOT one of them: it runs for real in every save below.
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    made = []

    def _make(area="Finance"):
        tag = uuid.uuid4().hex[:8]
        w = SimpleNamespace(tag=tag, key=None, live=2, owner=f"TST Owner {tag}",
                            owner_person=person(f"tst-owner-{tag}"))
        made.append(w)
        form = _form(tag)
        form["businessArea"] = area
        w.key = repo.create_draft(conn, form, actor="test")["policyKey"]
        repo.save_version(conn, w.key, form, base_version=1, intent="activate", actor="test", change_note="go")
        conn.cursor().execute(
            "INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) VALUES (%s, %s, 'test')",
            (w.owner, [w.owner_person.email]))
        fire(conn, w.key, BASE, version=w.live)
        [target] = [t for t in L.targets(conn.cursor()) if t.key == w.key]
        counts = L.scan(conn, NOW, only=[target])
        assert (counts["raised"], counts["errors"]) == (1, 0), counts
        [w.sid] = [c["id"] for c in L.for_policy(conn.cursor(), w.key)]
        return w

    yield _make
    c = conn.cursor()
    for w in made:
        c.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name LIKE %s", (f"TST % {w.tag}",))
        if w.key:
            c.execute("DELETE FROM proc.bp_agent_policy_second_review WHERE policy_key = %s", (w.key,))
            c.execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (w.key,))
            got = repo.get_policy(conn, w.key)
            if got["status"] != "retired":
                repo.retire(conn, w.key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


@pytest.fixture
def fin(make):
    return make("Finance")


def _at(conn, w, limit=None, **changes):
    """The live form with the limit (and anything else) changed."""
    form = copy.deepcopy(repo.get_policy(conn, w.key)["versions"][w.live - 1]["form"])
    if limit is not None:
        form["hidden"]["condition"]["all"][1]["value"] = limit
    form.update(changes)
    return form


def _draft(conn, w, form, *, base, actor="tst-saver"):
    return repo.save_version(conn, w.key, form, base_version=base, intent="draft", actor=actor,
                             change_note="TST draft")["version"]


def _activate(conn, w, form, *, base, actor="tst-saver"):
    return repo.save_version(conn, w.key, form, base_version=base, intent="activate", actor=actor,
                             change_note="TST activate")["version"]


def _refused(conn, w, form, *, base, actor="tst-saver"):
    before = _policy(conn, w)
    with pytest.raises(repo.NotReady) as exc:
        _activate(conn, w, form, base=base, actor=actor)
    assert _policy(conn, w) == before, "nothing written, the pointers unmoved"
    return exc.value.problems


def _policy(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT (SELECT count(*) FROM proc.bp_agent_policy_version WHERE policy_key = %s), status, "
                "live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s", (w.key, w.key))
    return cur.fetchone()


def _suggestion(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT status FROM proc.bp_agent_policy_suggestion WHERE suggestion_id = %s", (w.sid,))
    return cur.fetchone()[0]


def _reviews(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT version, suggestion_id, reviewed_by, reviewed_at, note FROM "
                "proc.bp_agent_policy_second_review WHERE policy_key = %s ORDER BY version", (w.key,))
    return cur.fetchall()


def _review(conn, w, version, who=REVIEWER, note=None):
    return SR.record(conn, w.key, version, principal=who, note=note, now=NOW)


def _code(conn, w, version, who):
    with pytest.raises(L.Refused) as exc:
        _review(conn, w, version, who)
    assert exc.value.message == SR.REFUSALS[exc.value.code]
    return exc.value.code


# ------------------------------------------------------------------ activation
def test_finance_activation_of_the_suggested_loosening_is_refused_without_a_review(conn, fin):
    form = _at(conn, fin, 800)
    # straight from the live version, with no draft saved: the suggestion is still open
    assert _refused(conn, fin, form, base=fin.live) == REQUIRED
    assert _suggestion(conn, fin) == "open", "the refused save marked nothing"
    v = _draft(conn, fin, form, base=fin.live)
    assert v == 3 and _suggestion(conn, fin) == "saved", "saving the loosening marks its suggestion saved"
    assert _refused(conn, fin, form, base=v) == REQUIRED, "a saved suggestion asks for the review just the same"
    assert _policy(conn, fin) == (3, "live", fin.live, 3) and _reviews(conn, fin) == []


def test_after_another_approver_reviews_it_activation_succeeds(conn, fin):
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    assert _review(conn, fin, v, note="  the evidence holds  ") == {
        "policyKey": fin.key, "version": v, "reviewedBy": "tst-reviewer", "reviewedAt": NOW.isoformat()}
    assert _reviews(conn, fin) == [(v, fin.sid, "tst-reviewer", NOW, "the evidence holds")]
    assert _code(conn, fin, v, person("tst-third")) == "already_reviewed"
    assert _activate(conn, fin, form, base=v) == v + 1
    assert _policy(conn, fin) == (4, "live", 4, 4)
    assert _reviews(conn, fin) == [(v, fin.sid, "tst-reviewer", NOW, "the evidence holds")], "the review stays on record"


def test_the_second_reviewer_may_be_the_one_who_activates(conn, fin):
    # the rules are about who reviews, not who presses Activate
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v, actor="tst-reviewer") == v + 1
    assert _policy(conn, fin)[1:] == ("live", v + 1, v + 1)


def test_a_review_of_one_draft_does_not_cover_a_different_condition(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    assert _refused(conn, fin, _at(conn, fin, 900), base=v) == REQUIRED, "reviewed at 800, activated at 900"
    assert _refused(conn, fin, _at(conn, fin, 600), base=v) == REQUIRED, "nor a smaller loosening nobody reviewed"
    # the same condition with another field edited is the change that was reviewed
    assert _activate(conn, fin, _at(conn, fin, 800, messageForAgent="TST reworded"), base=v) == v + 1


def test_a_review_of_one_draft_does_not_cover_the_draft_saved_after_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    newer = _draft(conn, fin, _at(conn, fin, 900), base=v)
    assert _refused(conn, fin, _at(conn, fin, 900), base=newer) == REQUIRED
    assert _code(conn, fin, v, person("tst-third")) == "not_latest", "the reviewed draft is no longer the latest"
    _review(conn, fin, newer)
    assert _activate(conn, fin, _at(conn, fin, 900), base=newer) == newer + 1


@pytest.mark.parametrize("state,limit", [("dismissed", 800), ("not_yet", 800), ("saved", 800),
                                         ("dismissed", 600), ("dismissed", 2000)])
def test_typing_the_loosening_by_hand_still_needs_the_review(conn, fin, state, limit):
    """Ruling Q3. The owner never used the card: they hid it (or left it) and typed the number.
    (The suggestion still `open` at the moment of activation is the first test above.)"""
    if state in ("dismissed", "not_yet"):
        L.act(conn, fin.sid, {"dismissed": "dismiss", "not_yet": "not_yet"}[state], actor="tst-buyer", now=NOW)
    form = _at(conn, fin, limit)
    v = _draft(conn, fin, form, base=fin.live)
    # the save's own bookkeeping: a hidden-for-now or open card that the save holds becomes saved; a
    # dismissed one stays dismissed. Whatever it became, the review is required.
    assert _suggestion(conn, fin) == {"dismissed": "dismissed"}.get(state, "saved")
    assert _refused(conn, fin, form, base=v) == REQUIRED
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v) == v + 1


def test_a_superseded_suggestion_or_none_at_all_asks_for_nothing(conn, make):
    one, two = make("Finance"), make("Finance")
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'superseded' WHERE suggestion_id = %s",
                          (one.sid,))
    assert _activate(conn, one, _at(conn, one, 800), base=one.live) == 3
    conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE suggestion_id = %s", (two.sid,))
    assert _activate(conn, two, _at(conn, two, 800), base=two.live) == 3


def test_operations_needs_no_second_review(conn, make):
    ops = make("Operations")
    form = _at(conn, ops, 800)
    v = _draft(conn, ops, form, base=ops.live)
    assert repo.get_policy(conn, ops.key)["secondReview"] == NONE
    assert _code(conn, ops, v, REVIEWER) == "not_needed"
    assert _activate(conn, ops, form, base=v) == v + 1 and _reviews(conn, ops) == []


def test_tightening_or_an_unrelated_edit_needs_no_review_in_finance(conn, fin):
    assert _activate(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live) == 3
    assert _activate(conn, fin, _at(conn, fin, 400), base=3) == 4
    assert _policy(conn, fin) == (4, "live", 4, 4) and _reviews(conn, fin) == []


def test_moving_the_policy_out_of_finance_in_the_same_request_does_not_avoid_the_review(conn, fin):
    moved = _at(conn, fin, 800, businessArea="Operations")
    assert _refused(conn, fin, moved, base=fin.live) == REQUIRED
    v = _draft(conn, fin, moved, base=fin.live)
    assert _refused(conn, fin, moved, base=v) == REQUIRED
    assert repo.get_policy(conn, fin.key)["secondReview"]["required"] is True


def test_moving_a_policy_into_finance_while_loosening_it_needs_the_review(conn, make):
    ops = make("Operations")
    assert _refused(conn, ops, _at(conn, ops, 800, businessArea="Finance"), base=ops.live) == REQUIRED


# ------------------------------------------------------------------ who may review
def test_the_person_who_saved_the_draft_cannot_review_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live, actor="tst-saver")
    assert _code(conn, fin, v, SAVER) == "own_change"
    assert _code(conn, fin, v, person("TST-Saver")) == "own_change", "the same subject, whatever its case"
    assert _code(conn, fin, v, person("another-subject", email="tst-saver")) == "own_change", \
        "saved under a name that is this caller's email"
    assert _reviews(conn, fin) == []


def test_the_owner_cannot_review_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    assert _code(conn, fin, v, fin.owner_person) == "owner"
    assert _reviews(conn, fin) == []
    _review(conn, fin, v)                                   # someone linked to nobody may


def test_naming_another_owner_in_the_draft_does_not_let_the_owner_review(conn, fin):
    other = f"TST Other Owner {fin.tag}"
    conn.cursor().execute("INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) "
                          "VALUES (%s, %s, 'test')", (other, ["tst-someone-else@example.test"]))
    v = _draft(conn, fin, _at(conn, fin, 800, owner=other), base=fin.live)
    assert _code(conn, fin, v, fin.owner_person) == "owner", "the owner of the version in force"
    assert _code(conn, fin, v, person("tst-someone-else")) == "owner", "and the owner the draft names"
    assert _reviews(conn, fin) == []


def test_an_owner_linked_to_nobody_means_no_review_can_be_recorded(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    conn.cursor().execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = %s", (fin.owner,))
    assert _code(conn, fin, v, REVIEWER) == "owner_unmapped"
    assert _reviews(conn, fin) == [], "nobody is let through because the rule cannot be evaluated"


def test_a_review_is_refused_when_none_is_needed_or_the_version_is_not_the_latest_draft(conn, fin):
    assert _code(conn, fin, fin.live, REVIEWER) == "not_latest", "the latest version is live, not a draft"
    v = _draft(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live)
    assert _code(conn, fin, v, REVIEWER) == "not_needed", "the condition is unchanged"
    assert _code(conn, fin, fin.live, REVIEWER) == "not_latest"
    assert _code(conn, fin, v + 5, REVIEWER) == "not_latest"
    tighter = _draft(conn, fin, _at(conn, fin, 400), base=v)
    assert _code(conn, fin, tighter, REVIEWER) == "not_needed", "a tightening"
    with pytest.raises(L.Refused) as exc:
        SR.record(conn, "TST-nosuch", 1, principal=REVIEWER, note=None, now=NOW)
    assert exc.value.code == "not_found"
    assert _reviews(conn, fin) == []


# ------------------------------------------------------------------ fail closed
def test_activation_is_refused_when_the_suggestions_cannot_be_read(conn, fin, monkeypatch):
    form = _at(conn, fin, 400)                               # a tightening: it would need no review at all
    real = SR._SUGGESTIONS_SQL
    monkeypatch.setattr(SR, "_SUGGESTIONS_SQL", "SELECT suggestion_id, status, proposal FROM proc.bp_tst_no_such_table "
                                                "WHERE policy_key = %s")
    assert _refused(conn, fin, form, base=fin.live) == UNAVAILABLE
    assert conn.autocommit is True
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "no draft: the page has nothing to work out"
    monkeypatch.setattr(SR, "_SUGGESTIONS_SQL", real)
    assert _activate(conn, fin, form, base=fin.live) == 3, "the connection was left usable"


def test_activation_is_refused_when_the_business_area_cannot_be_read(conn, fin, monkeypatch):
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    _review(conn, fin, v)                                    # even a recorded review does not help
    monkeypatch.setattr(SR, "_SETTING_SQL", "SELECT second_reviewer FROM proc.bp_tst_no_such_table WHERE area_name = %s")
    assert _refused(conn, fin, form, base=v) == UNAVAILABLE
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}, "the page fails closed too"


def test_a_stored_suggestion_that_cannot_be_read_refuses_the_activation(conn, fin):
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET proposal = '{\"kind\": \"threshold\"}'::jsonb "
                          "WHERE suggestion_id = %s", (fin.sid,))
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == UNAVAILABLE


def test_no_review_is_recorded_when_the_decider_map_cannot_be_read(conn, fin, monkeypatch):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)

    def boom(c):
        raise RuntimeError("the map cannot be read")
    monkeypatch.setattr(SR.deciders, "load_map", boom)
    assert _code(conn, fin, v, REVIEWER) == "second_review_unavailable"
    assert _reviews(conn, fin) == [] and conn.autocommit is True


# ------------------------------------------------------------------ the policy page
def test_the_policy_page_says_a_review_is_required_and_then_who_did_it(conn, fin):
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "nothing newer than the live version"
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    _review(conn, fin, v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v,
                                                              "reviewedBy": "tst-reviewer",
                                                              "reviewedAt": NOW.isoformat()}
    _activate(conn, fin, form, base=v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "it is live: nothing waits for a review"


def test_the_policy_page_asks_for_nothing_on_a_draft_that_does_not_loosen(conn, fin):
    _draft(conn, fin, _at(conn, fin, 400), base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE


# ------------------------------------------------------------------ through the whole app
def _hdr(sub):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": f"{sub}@example.test", "X-User-Groups": "[]"}


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(R, "_role_of", lambda p: ROLES.get(p.subject, "Nobody"))
    return TestClient(app)


def test_the_endpoint_needs_the_approver_role(client, conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    r = client.post(f"/agent-policies/{fin.key}/second-review", json={"version": v}, headers=_hdr("tst-buyer"))
    assert r.status_code == 403 and _reviews(conn, fin) == []


def test_the_whole_flow_through_the_app(client, conn, fin):
    form = _at(conn, fin, 800)
    versions, review = f"/agent-policies/{fin.key}/versions", f"/agent-policies/{fin.key}/second-review"
    r = client.post(versions, headers=_hdr("tst-saver"),
                    json={"form": form, "baseVersion": fin.live, "intent": "draft", "changeNote": "TST"})
    assert r.status_code == 200 and r.json()["version"] == 3, r.text
    activate = {"form": form, "baseVersion": 3, "intent": "activate", "changeNote": "TST"}
    r = client.post(versions, headers=_hdr("tst-saver"), json=activate)
    assert r.status_code == 422 and r.json() == {"problems": REQUIRED}, "refused, and the message arrives readable"
    page = client.get(f"/agent-policies/{fin.key}", headers=_hdr("tst-saver")).json()
    assert page["secondReview"] == {"required": True, "version": 3, "reviewedBy": None, "reviewedAt": None}
    r = client.post(review, json={"version": 3}, headers=_hdr("tst-saver"))
    assert r.status_code == 409 and r.json() == {"code": "own_change", "message": SR.REFUSALS["own_change"]}
    r = client.post(review, json={"version": 2}, headers=_hdr("tst-reviewer"))
    assert r.status_code == 409 and r.json()["code"] == "not_latest"
    r = client.post(review, json={"version": 3, "note": "TST looked"}, headers=_hdr("tst-reviewer"))
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"policyKey", "version", "reviewedBy", "reviewedAt"}
    assert (r.json()["policyKey"], r.json()["version"], r.json()["reviewedBy"]) == (fin.key, 3, "tst-reviewer")
    [(version, sid, by, _at_, note)] = _reviews(conn, fin)
    assert (version, sid, by, note) == (3, fin.sid, "tst-reviewer", "TST looked")
    page = client.get(f"/agent-policies/{fin.key}", headers=_hdr("tst-saver")).json()
    assert (page["secondReview"]["reviewedBy"], page["secondReview"]["reviewedAt"]) == ("tst-reviewer", r.json()["reviewedAt"])
    r = client.post(versions, headers=_hdr("tst-saver"), json=activate)
    assert r.status_code == 200 and r.json()["version"] == 4, r.text
    assert _policy(conn, fin) == (4, "live", 4, 4)
    assert client.post("/agent-policies/ZZZ-999999/second-review", json={"version": 1},
                       headers=_hdr("tst-reviewer")).status_code == 404
