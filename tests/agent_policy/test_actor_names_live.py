"""People are shown by name (stage 5b, B1), on bp_testdb. Needs PROCWISE_TEST_LIVE_DB=1.

A second review and a reversal keep the sign-in email of whoever recorded them beside the sign-in
id; the policy page and the activity list show the email when there is one, else the id. Rows
written before the email column existed (NULL) still read. Nothing records the email of whoever
decided an approval, so "Approved by" is the id, or the system's words for a `system:*` decider.

Data as the stage 5 live tests make it: TST-<hex>a keys for the direct tests, a real "TST …" policy
(retired at teardown) where the page or an endpoint is read. Reversal and firing rows are
append-only BY DESIGN and stay; review rows are removed by the fixture.
"""
import os

import pytest

from api.auth import Principal
from repositories import agent_policy_repo as repo
from services.agent_policy import approval_views as V
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from tests.agent_policy import test_learning_endpoints_live as E
from tests.agent_policy import test_learning_reversal_live as RV
from tests.agent_policy import test_second_review_live as S
from tests.agent_policy.test_learning_reversal_live import api, policy  # noqa: F401
from tests.agent_policy.test_learning_scan_live import BASE, NOW, _reverse, fire, world  # noqa: F401
from tests.agent_policy.test_second_review_live import conn, fin, make  # noqa: F401

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

WHY = "TST the refund was clawed back"


def _reversal_email(conn, firing_id):
    cur = conn.cursor()
    cur.execute("SELECT reversed_by, reversed_by_email FROM proc.bp_policy_firing_reversal WHERE firing_id = %s",
                (firing_id,))
    return cur.fetchone()


def _review_email(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT reviewed_by, reviewed_by_email FROM proc.bp_agent_policy_second_review WHERE policy_key = %s",
                (key,))
    return cur.fetchall()


# ------------------------------------------------------------------ reversals
def test_a_reversal_keeps_the_email_trimmed_and_the_activity_list_shows_it(world):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    out = L.record_reversal(conn, fid, actor="tst-approver", email="  Ana@Example.test ", reason=WHY, now=NOW)
    assert set(out) == {"firingId", "reversedBy", "reversedAt", "reason"}, "the answer names nobody new"
    assert _reversal_email(conn, fid) == ("tst-approver", "Ana@Example.test")
    [row] = V.policy_firings(conn, t.key, 50)
    assert row["reversed"] == {"by": "tst-approver", "byName": "Ana@Example.test", "at": NOW.isoformat(),
                               "reason": WHY}


@pytest.mark.parametrize("email", [None, "", "   "])
def test_a_reversal_by_a_sign_in_without_an_email_stores_none_and_shows_the_id(world, email):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    L.record_reversal(conn, fid, actor="tst-approver", email=email, reason=WHY, now=NOW)
    assert _reversal_email(conn, fid) == ("tst-approver", None)
    [row] = V.policy_firings(conn, t.key, 50)
    assert (row["reversed"]["by"], row["reversed"]["byName"]) == ("tst-approver", "tst-approver")


def test_a_reversal_written_before_the_email_column_still_reads(world):
    conn, t = world
    [fid, other] = fire(conn, t.key, [600, 610])
    _reverse(conn, fid)                              # as stage 5 wrote it: no email
    assert _reversal_email(conn, fid) == ("tst", None)
    rows = {f["id"]: f for f in V.policy_firings(conn, t.key, 50)}
    assert (rows[fid]["reversed"]["by"], rows[fid]["reversed"]["byName"]) == ("tst", "tst")
    assert rows[other]["reversed"] is None, "no reversal: no id and no name"


def test_the_reversal_endpoint_stores_the_callers_email_and_the_list_arrives_readable(api, policy):
    at_600, at_610 = policy.ids[BASE.index(600)], policy.ids[BASE.index(610)]
    r = RV._post(api, at_600)
    assert r.status_code == 201, r.text
    assert "example.test" not in r.text, "the POST answer carries the id, as before, and no address"
    assert _reversal_email(policy.conn, at_600) == ("tst-approver", "tst-approver@example.test")
    no_email = {**E.APPROVER, "X-User-Email": "  "}
    assert RV._post(api, at_610, headers=no_email).status_code == 201
    assert _reversal_email(policy.conn, at_610) == ("tst-approver", None)
    # through the whole app (the output filter passes this read through): the name is readable
    rows = {f["id"]: f for f in api.client.get(f"/agent-policies/{policy.key}/firings?limit=50",
                                               headers=E.VIEWER).json()["firings"]}
    assert rows[at_600]["reversed"]["byName"] == "tst-approver@example.test"
    assert rows[at_610]["reversed"]["byName"] == "tst-approver"
    assert all(f["decidedByName"] == f["decidedBy"] for f in rows.values()), "people: no email is recorded"
    assert {f["decidedBy"] for f in rows.values()} == {"person-0", "person-1", "person-2"}


# ------------------------------------------------------------------ approved by
def test_recent_decisions_name_the_decider_and_word_the_system(world):
    conn, t = world
    [person] = fire(conn, t.key, [600])
    [timed_out] = fire(conn, t.key, [610], result="timed_out", by="system:timeout")
    [grouped] = fire(conn, t.key, [620], by="system:group")
    paused = RV._firing(conn, t.key, "approve", "paused_for_approval", by=None)
    rows = {f["id"]: f for f in V.policy_firings(conn, t.key, 50)}
    assert (rows[person]["decidedBy"], rows[person]["decidedByName"]) == ("person-0", "person-0")
    assert (rows[timed_out]["decidedBy"], rows[timed_out]["decidedByName"]) == ("system:timeout", "Timeout")
    assert (rows[grouped]["decidedBy"], rows[grouped]["decidedByName"]) == ("system:group", "By the system")
    assert (rows[paused]["decidedBy"], rows[paused]["decidedByName"]) == (None, None)


# ------------------------------------------------------------------ second reviews
def test_a_second_review_keeps_the_email_and_the_page_shows_it(conn, fin):
    v = S._draft(conn, fin, S._at(conn, fin, 800), base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {
        "required": True, "version": v, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}
    who = Principal(subject="tst-reviewer", email="  tst-reviewer@example.test ", claims={"cognito:groups": []})
    out = SR.record(conn, fin.key, v, principal=who, note=None, now=NOW)
    assert set(out) == {"policyKey", "version", "reviewedBy", "reviewedAt"}, "the answer names nobody new"
    assert _review_email(conn, fin.key) == [("tst-reviewer", "tst-reviewer@example.test")]
    assert repo.get_policy(conn, fin.key)["secondReview"] == {
        "required": True, "version": v, "reviewedBy": "tst-reviewer",
        "reviewedByName": "tst-reviewer@example.test", "reviewedAt": NOW.isoformat()}


def test_a_second_review_by_a_sign_in_without_an_email_stores_none_and_shows_the_id(conn, fin):
    # the owner linked by group, not by email: a sign-in without an email can then be told apart
    conn.cursor().execute("UPDATE proc.bp_policy_decider_map SET emails = '{}', groups = %s WHERE decider_name = %s",
                          ([f"TST-G-{fin.tag}"], fin.owner))
    v = S._draft(conn, fin, S._at(conn, fin, 800), base=fin.live)
    for blank in ("   ", None):
        who = Principal(subject="tst-reviewer", email=blank, claims={"cognito:groups": []})
        conn.cursor().execute("DELETE FROM proc.bp_agent_policy_second_review WHERE policy_key = %s", (fin.key,))
        SR.record(conn, fin.key, v, principal=who, note=None, now=NOW)
        assert _review_email(conn, fin.key) == [("tst-reviewer", None)]
        review = repo.get_policy(conn, fin.key)["secondReview"]
        assert (review["reviewedBy"], review["reviewedByName"]) == ("tst-reviewer", "tst-reviewer")


def test_a_second_review_written_before_the_email_column_still_reads_and_still_lets_it_activate(conn, fin):
    form = S._at(conn, fin, 800)
    v = S._draft(conn, fin, form, base=fin.live)
    conn.cursor().execute("INSERT INTO proc.bp_agent_policy_second_review (policy_key, version, suggestion_id, "
                          "reviewed_by, reviewed_at) VALUES (%s, %s, %s, 'tst-old-reviewer', %s)",
                          (fin.key, v, fin.sid, NOW))                # as stage 5 wrote it: no email
    review = repo.get_policy(conn, fin.key)["secondReview"]
    assert (review["reviewedBy"], review["reviewedByName"]) == ("tst-old-reviewer", "tst-old-reviewer")
    assert S._activate(conn, fin, form, base=v) == v + 1, "the review still covers the draft"


def test_the_policy_page_through_the_app_shows_the_reviewer_by_name(conn, fin, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    from api.routers import agent_policies as R
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(R, "_role_of", lambda p: S.ROLES.get(p.subject, "Nobody"))
    client = TestClient(app)
    v = S._draft(conn, fin, S._at(conn, fin, 800), base=fin.live)
    r = client.post(f"/agent-policies/{fin.key}/second-review", json={"version": v}, headers=S._hdr("tst-reviewer"))
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"policyKey", "version", "reviewedBy", "reviewedAt"} and "example.test" not in r.text
    assert _review_email(conn, fin.key) == [("tst-reviewer", "tst-reviewer@example.test")]
    page = client.get(f"/agent-policies/{fin.key}", headers=S._hdr("tst-buyer")).json()
    assert (page["secondReview"]["reviewedBy"], page["secondReview"]["reviewedByName"]) == \
        ("tst-reviewer", "tst-reviewer@example.test"), "readable after the output filter (an exempt read)"
