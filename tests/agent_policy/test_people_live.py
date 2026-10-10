"""A sign-in id resolves to a name, on bp_testdb (stage 5b, Task B7). Needs PROCWISE_TEST_LIVE_DB=1.

test_approval_endpoints_live's world (a made-up policy key TST-<digits>, two decider names, real
approval cases). Every person here has a sign-in id of this test's own ("tst-p-<tag>-…"); their
rows in proc.bp_agent_policy_person are removed by the fixture.
"""
import os
from datetime import timedelta

import pytest

from api.auth import Principal
from services.agent_policy import approval_views as V
from services.agent_policy import approvals as A
from services.agent_policy import people as P
from tests.agent_policy import test_approval_endpoints_live as E
from tests.agent_policy.test_approval_endpoints_live import client, conn, world  # noqa: F401 - fixtures

pytestmark = [pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required"),
              pytest.mark.people]


@pytest.fixture
def who(conn, world):
    """This test's people: sign-in ids nobody else uses."""
    prefix = f"tst-p-{world.tag}"
    yield lambda name: f"{prefix}-{name}"
    conn.cursor().execute("DELETE FROM proc.bp_agent_policy_person WHERE subject LIKE %s", (prefix + "-%",))


def stored(conn, subject):
    cur = conn.cursor()
    cur.execute("SELECT email, first_seen_at, last_seen_at FROM proc.bp_agent_policy_person WHERE subject = %s",
                (subject,))
    return cur.fetchone()


def test_remember_keeps_the_email_and_a_blank_one_never_erases_it(conn, who):
    sub = who("a")
    assert stored(conn, sub) is None
    P.remember(conn, Principal(subject=sub, email="  First@Example.test ", claims={}))
    email, first, last = stored(conn, sub)
    assert email == "First@Example.test" and first == last
    # a sign-in that carries no email: the stored one stays, untouched
    for blank in (None, "", "   "):
        P.remember(conn, Principal(subject=sub, email=blank, claims={}))
        assert stored(conn, sub) == (email, first, last), repr(blank)
    # a new address replaces it; when they were first seen does not move
    P.remember(conn.cursor(), Principal(subject=sub, email="second@example.test", claims={}))
    email2, first2, last2 = stored(conn, sub)
    assert (email2, first2) == ("second@example.test", first) and last2 > last
    assert P.names(conn.cursor(), [sub, who("nobody"), "system:timeout", None]) == {sub: "second@example.test"}
    assert conn.autocommit is True


def test_a_long_email_is_capped_and_a_failed_write_changes_nothing_and_raises_nothing(conn, who, monkeypatch):
    sub = who("long")
    P.remember(conn, Principal(subject=sub, email="a" * 400 + "@example.test", claims={}))
    assert len(stored(conn, sub)[0]) == 320
    before = stored(conn, sub)
    monkeypatch.setattr(P, "_REMEMBER_SQL", "INSERT INTO proc.bp_agent_policy_person_no_such (subject, email) "
                                            "VALUES (%s, %s)")
    assert P.remember(conn, Principal(subject=sub, email="other@example.test", claims={})) is None
    assert stored(conn, sub) == before
    # inside a transaction the failure is undone on its own: the caller's work survives it
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute("UPDATE proc.bp_agent_policy_person SET email = 'kept@example.test' WHERE subject = %s", (sub,))
        P.remember(cur, Principal(subject=sub, email="other@example.test", claims={}))
        cur.execute("SELECT 1")                       # the transaction is still usable
        conn.commit()
    finally:
        conn.autocommit = True
    assert stored(conn, sub)[0] == "kept@example.test"


def test_a_write_through_the_app_remembers_the_caller_and_a_get_does_not_write(client, conn, world, who):
    _did, _fid, nid = E._open(conn, world)
    sub, reader = who("l1"), who("reader")
    l1 = E._hdr(sub, f"  {world.l1_email} ")
    # reads first: nothing is written, for anybody
    assert client.get("/agent-policies/notifications?mine=1", headers=l1).status_code == 200
    assert client.get("/agent-policies/approvals", headers=E._hdr(reader, "reader@example.test")).status_code == 200
    assert client.get(f"/agent-policies/{world.key}/firings", headers=l1).status_code == 200
    assert stored(conn, sub) is None and stored(conn, reader) is None
    # a write: remembered, trimmed
    assert client.post(f"/agent-policies/notifications/{nid}/read", headers=l1).json() == {"id": nid, "read": True}
    email, first, last = stored(conn, sub)
    assert email == world.l1_email
    # and a read afterwards does not touch the row (not even "last seen")
    assert client.get("/agent-policies/notifications?mine=1", headers=l1).status_code == 200
    assert client.get("/agent-policies/approvals", headers=l1).status_code == 200
    assert stored(conn, sub) == (email, first, last)
    # a refused write (someone else's notification) is still a write by a signed-in person
    other = who("other")
    assert client.post(f"/agent-policies/notifications/{nid}/read",
                       headers=E._hdr(other, "other@example.test")).status_code == 404
    assert stored(conn, other)[0] == "other@example.test"


def _spy_on_names(monkeypatch):
    asked = []
    real = P.names
    monkeypatch.setattr(V.people, "names", lambda cur, subjects: asked.append(list(subjects)) or real(cur, subjects))
    return asked


def test_approved_by_and_the_cards_history_show_the_email_for_a_known_id_and_the_id_for_an_unknown_one(
        client, conn, world, who, monkeypatch):
    l1, l2 = who("l1"), who("l2")
    l1_hdr, l2_hdr = E._hdr(l1, world.l1_email), E._hdr(l2, "", [world.l2_group])
    # one request approved by a person whose sign-in carries an email: remembered by that write
    known, known_firing, _ = E._open(conn, world)
    r = client.post(f"/agent-policies/approvals/{known}/decide", json={"verb": "approve"}, headers=l1_hdr)
    assert r.status_code == 200 and r.json()["result"] == "approved", r.text
    # another, escalated to the second level, rejected by someone linked by group whose sign-in
    # carries no email: there is nothing to remember, so they stay their id
    unknown, unknown_firing, _ = E._open(conn, world)
    assert A.sweep(conn, E.NOW + timedelta(days=30), decision_ids=[unknown])["escalated"] == 1
    r = client.post(f"/agent-policies/approvals/{unknown}/decide", json={"verb": "reject", "reason": "TST no"},
                    headers=l2_hdr)
    assert r.status_code == 200 and r.json()["result"] == "rejected", r.text
    assert stored(conn, l1)[0] == world.l1_email and stored(conn, l2) is None

    asked = _spy_on_names(monkeypatch)
    one = client.get(f"/agent-policies/approvals/{known}", headers=l1_hdr)
    assert one.status_code == 200, one.text
    hist = one.json()["history"]
    assert [(d["verb"], d["by"], d["byName"]) for d in hist["decisions"]] == [("approve", l1, world.l1_email)]
    assert [(f["decidedBy"], f["decidedByName"]) for f in hist["firings"]] == [(l1, world.l1_email)]
    assert len(asked) == 1, "one names query for the whole answer"
    assert "[withheld]" not in one.text, "the output filter left the address readable"

    del asked[:]
    other = client.get(f"/agent-policies/approvals/{unknown}", headers=l2_hdr).json()["history"]
    assert [(d["verb"], d["by"], d["byName"]) for d in other["decisions"]] == [("reject", l2, l2)]
    assert [(f["decidedBy"], f["decidedByName"]) for f in other["firings"]] == [(l2, l2)]
    assert len(asked) == 1

    # "Approved by <email>" in the policy's activity list, to someone who only reads
    del asked[:]
    got = client.get(f"/agent-policies/{world.key}/firings", headers=E._hdr(who("viewer"), "v@example.test"))
    assert got.status_code == 200, got.text
    rows = {f["id"]: (f["decidedBy"], f["decidedByName"]) for f in got.json()["firings"]}
    assert rows == {known_firing: (l1, world.l1_email), unknown_firing: (l2, l2)}
    assert len(asked) == 1 and "[withheld]" not in got.text
    assert stored(conn, who("viewer")) is None, "reading the list remembered nobody"


def test_a_timeout_keeps_its_words_and_is_never_looked_up(client, conn, world, who, monkeypatch):
    did, _fid, _ = E._open(conn, world)
    assert A.sweep(conn, E.NOW + timedelta(days=30), decision_ids=[did])["escalated"] == 1
    assert A.sweep(conn, E.NOW + timedelta(days=60), decision_ids=[did])["rejected"] == 1
    asked = _spy_on_names(monkeypatch)
    hist = client.get(f"/agent-policies/approvals/{did}",
                      headers=E._hdr(who("admin"), "a@example.test", ["PROCWISE_ADMIN"])).json()["history"]
    assert [(d["by"], d["byName"], d["timeout"]) for d in hist["decisions"]] == [("system:timeout", "Timeout", True)]
    assert [(f["decidedBy"], f["decidedByName"]) for f in hist["firings"]] == [("system:timeout", "Timeout")]
    assert len(asked) == 1
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_agent_policy_person WHERE subject LIKE 'system:%'")
    assert cur.fetchone()[0] == 0


def test_remembering_that_fails_does_not_fail_the_persons_action(client, conn, world, who, monkeypatch):
    _did, _fid, nid = E._open(conn, world)
    sub = who("l1")
    monkeypatch.setattr(P, "_REMEMBER_SQL", "INSERT INTO proc.bp_agent_policy_person_no_such (subject, email) "
                                            "VALUES (%s, %s)")
    r = client.post(f"/agent-policies/notifications/{nid}/read", headers=E._hdr(sub, world.l1_email))
    assert r.status_code == 200 and r.json() == {"id": nid, "read": True}
    cur = conn.cursor()
    cur.execute("SELECT read_by FROM proc.bp_policy_notification WHERE notification_id = %s", (nid,))
    assert cur.fetchone()[0] == [sub], "the action itself went through"
    assert stored(conn, sub) is None
