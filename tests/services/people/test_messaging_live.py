"""Messaging against the real database (PROCWISE_TEST_LIVE_DB=1). Uses subjects prefixed zm_ and removes them."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="needs the live database")

from services.db import get_conn
from services.people import messaging as m
from services.people.admin import PeopleError
from services.people.wiring import drain_once

A, B, C, D = "zm_alice", "zm_bob", "zm_carol", "zm_dave"


@pytest.fixture()
def db():
    with get_conn() as conn:
        cur = conn.cursor()
        def clean():
            cur.execute("DELETE FROM proc.bp_notification WHERE user_subject LIKE 'zm\\_%'")
            cur.execute("DELETE FROM proc.bp_domain_event WHERE actor_id LIKE 'zm\\_%'")
            cur.execute("DELETE FROM proc.bp_conversation WHERE created_by LIKE 'zm\\_%'")
            cur.execute("DELETE FROM proc.bp_person WHERE subject LIKE 'zm\\_%'")
        clean()
        for s, mgr in ((A, None), (B, A), (C, None), (D, None)):
            cur.execute("INSERT INTO proc.bp_person (subject, display_name, manager_subject, email) VALUES (%s,%s,%s,%s)",
                        (s, s[3:].title(), mgr, f"{s[3:]}@example.com"))
        cur.execute("UPDATE proc.bp_person SET is_active=false WHERE subject=%s", (D,))
        yield conn
        clean()


def inbox(conn, who):
    with conn.cursor() as cur:
        cur.execute("SELECT reason, title FROM proc.bp_notification WHERE user_subject=%s ORDER BY notification_id", (who,))
        return cur.fetchall()


def refused(status, fn, *a, **k):
    with pytest.raises(PeopleError) as e:
        fn(*a, **k)
    assert e.value.status == status
    return e.value.detail


def test_direct_is_one_per_pair_and_private(db):
    c1 = m.open_direct(db, A, B)["id"]
    assert m.open_direct(db, B, A)["id"] == c1
    refused(422, m.open_direct, db, A, A)
    refused(422, m.open_direct, db, A, D)           # inactive
    refused(422, m.open_direct, db, A, "zm_ghost")
    m.send(db, A, c1, "hello")
    refused(404, m.list_messages, db, C, c1, None, 10)      # outsider: looks like it does not exist
    refused(404, m.send, db, C, c1, "let me in")
    refused(404, m.get_conversation, db, C, c1)
    refused(404, m.mark_read, db, C, c1, None)
    assert [x["body"] for x in m.list_messages(db, B, c1, None, 10)["messages"]] == ["hello"]


def test_unread_counts_and_read(db):
    c1 = m.open_direct(db, A, B)["id"]
    for t in ("one", "two", "three"):
        m.send(db, A, c1, t)
    assert m.list_conversations(db, B)["unread_total"] == 3
    assert m.list_conversations(db, A)["unread_total"] == 0          # own messages are read
    first = m.list_messages(db, B, c1, None, 10)["messages"][-1]["id"]
    m.mark_read(db, B, c1, first)
    assert m.list_conversations(db, B)["unread_total"] == 2
    m.mark_read(db, B, c1, None)
    assert m.list_conversations(db, B)["unread_total"] == 0


def test_message_notifies_others_without_text_and_not_the_sender(db):
    cid = m.open_group(db, A, "Team", [B, C])["id"]
    m.send(db, A, cid, "secret plans")
    drain_once()
    assert inbox(db, A) == []
    for who in (B, C):
        rows = inbox(db, who)
        assert rows == [("explicit", "New message from Alice")]
        assert "secret" not in rows[0][1]


def test_mute_silences_messages_but_not_mentions(db):
    cid = m.open_group(db, A, "Team", [B, C])["id"]
    m.set_muted(db, B, cid, True)
    m.send(db, A, cid, "ordinary")
    m.send(db, A, cid, "hey @[Bob](zm_bob)")
    drain_once()
    assert [r[0] for r in inbox(db, B)] == ["mentioned"]
    assert [r[0] for r in inbox(db, C)] == ["explicit", "explicit"]


def test_mentions_rules(db):
    cid = m.open_group(db, A, "Team", [B])["id"]
    refused(422, m.send, db, A, cid, "hi @[Carol](zm_carol)")         # outsider in a group: refused
    refused(422, m.send, db, A, cid, "hi", mentions=["zm_ghost"])
    out = m.send(db, A, cid, "hi @[Bob](zm_bob) @[Alice](zm_alice)")
    assert out["mentions"] == ["zm_bob"]                               # you cannot mention yourself
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_message WHERE conversation_id=%s", (cid,))
        assert cur.fetchone()[0] == 1                                  # refused sends stored nothing


def test_entity_thread_is_invite_only_and_a_mention_invites(db):
    t = m.open_entity_thread(db, A, "PO", "PO-1001", None)
    assert t["created"] is True
    assert m.open_entity_thread(db, A, "po", "PO-1001", None) == {**t, "created": False}
    assert "ask someone" in refused(409, m.open_entity_thread, db, C, "po", "PO-1001", None)
    refused(422, m.open_entity_thread, db, A, "spaceship", "1", None)
    refused(404, m.list_messages, db, C, t["id"], None, 10)
    m.send(db, A, t["id"], "please look @[Carol](zm_carol)")           # participant's mention invites
    assert [x["body"] for x in m.list_messages(db, C, t["id"], None, 10)["messages"]] == ["please look @[Carol](zm_carol)"]
    drain_once()
    assert ("mentioned", "Alice mentioned you") in inbox(db, C)


def test_add_leave_and_direct_cannot_grow(db):
    d = m.open_direct(db, A, B)["id"]
    refused(422, m.add_participant, db, A, d, C)
    g = m.open_group(db, A, "G", [B])["id"]
    refused(404, m.add_participant, db, C, g, D)                       # outsider cannot add
    refused(422, m.add_participant, db, A, g, D)                       # inactive
    m.add_participant(db, A, g, C)
    m.leave(db, C, g)
    refused(404, m.list_messages, db, C, g, None, 10)
    m.send(db, A, g, "after C left")
    drain_once()
    assert inbox(db, C) == []                                          # a leaver is not notified


def test_edit_delete_and_replies(db):
    cid = m.open_direct(db, A, B)["id"]
    mid = m.send(db, A, cid, "orginal")["id"]
    refused(403, m.edit, db, B, mid, "tampered")
    refused(403, m.delete, db, B, mid)
    refused(404, m.edit, db, C, mid, "x")
    m.edit(db, A, mid, "original")
    reply = m.send(db, B, cid, "ok", reply_to_id=mid)["id"]
    refused(422, m.send, db, B, cid, "bad reply", reply_to_id=999999999)
    m.delete(db, A, mid)
    msgs = {x["id"]: x for x in m.list_messages(db, B, cid, None, 10)["messages"]}
    assert msgs[mid]["deleted"] is True and msgs[mid]["body"] == "" and msgs[reply]["reply_to_id"] == mid
    refused(409, m.edit, db, A, mid, "undelete")


def test_pagination(db):
    cid = m.open_direct(db, A, B)["id"]
    for i in range(5):
        m.send(db, A, cid, f"m{i}")
    p1 = m.list_messages(db, B, cid, None, 2)
    assert [x["body"] for x in p1["messages"]] == ["m4", "m3"] and p1["next_before_id"]
    p2 = m.list_messages(db, B, cid, p1["next_before_id"], 2)
    assert [x["body"] for x in p2["messages"]] == ["m2", "m1"]


def test_suggestions_are_labelled_and_search_finds_people(db):
    s = m.suggest_recipients(db, B, "caro")
    labels = [g["label"] for g in s["suggested"]]
    assert labels[0] == "Your manager" and s["suggested"][0]["people"][0]["subject"] == A
    assert s["suggested"][0]["tier"] == "asserted"
    assert [p["subject"] for p in s["search"]] == [C]
    assert m.suggest_recipients(db, B, "dave")["search"] == []          # inactive people are not offered


def test_someone_who_leaves_before_the_worker_runs_is_not_notified(db):
    cid = m.open_group(db, A, "Team", [B, C])["id"]
    m.send(db, A, cid, "queued while C was still in")
    m.leave(db, C, cid)                       # leaves after the event was queued, before it is processed
    drain_once()
    assert inbox(db, C) == [] and len(inbox(db, B)) == 1
