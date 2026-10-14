"""The owner is told when learning raises a suggestion (stage 5b, Task B4). Needs PROCWISE_TEST_LIVE_DB=1.

The scan is always scoped: targets are injected (keys TST-<hex>a, never real policies), as in
test_learning_scan_live, whose `world`, `fire` and base data (33 approvals, 510..830: one band of
30, to = 800) are reused. Each test's owner name is its own ("TST Owner <hex>"); every
notification row and decider-map row made here is removed by the fixture. Firing rows stay.

The notification table needs a row to hang on (`ck_bp_policy_notification_target`: a firing, or a
conflict link), so a suggestion's notification names the newest decision of its evidence.
"""
import logging
import os
import uuid
from dataclasses import replace

import pytest

from api.auth import Principal
from services.agent_policy import approval_views as AV
from services.agent_policy import gate
from services.agent_policy import learning as L
from tests.agent_policy.test_learning_scan_live import (BASE, CHANGE_500, NOW, _marked_sensitive, _set_status,
                                                        by_change, fire, row)
from tests.agent_policy.test_learning_scan_live import world as _scan_world  # noqa: F401 - the fixture

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
PLAIN = "Raise the limit on Refund amount from 500 USD to 800 USD?"


@pytest.fixture
def world(_scan_world):
    conn, target = _scan_world
    owner = f"TST Owner {uuid.uuid4().hex[:8]}"
    target = replace(target, form={**target.form, "owner": owner})
    yield conn, target, owner
    cur = conn.cursor()
    cur.execute("DELETE FROM proc.bp_policy_notification WHERE recipient LIKE %s", (owner + "%",))
    cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = %s", (owner,))


def notes(conn, owner):
    cur = conn.cursor()
    cur.execute("SELECT n.recipient, n.message, n.link, f.policy_key, n.firing_id FROM proc.bp_policy_notification n "
                "LEFT JOIN proc.bp_policy_firing f ON f.firing_id = n.firing_id "
                "WHERE n.recipient LIKE %s ORDER BY n.notification_id", (owner + "%",))
    return cur.fetchall()


def _sid(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT suggestion_id FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (key,))
    [(sid,)] = cur.fetchall()
    return sid


def test_a_raised_card_tells_the_owner_once_in_these_words(world):
    conn, t, owner = world
    ids = fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    sid = _sid(conn, t.key)
    [(recipient, message, link, policy_key, firing_id)] = notes(conn, owner)
    assert (recipient, link, policy_key) == (owner, f"suggestion:{sid}", t.key)
    assert message == f"Learning suggests a change to {t.key}: {PLAIN}"
    assert firing_id in ids[:30], "it hangs on a decision of the evidence (510..800), never on one outside it"


def test_a_plain_refresh_of_an_open_card_tells_nobody_again(world):
    conn, t, owner = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    for _ in range(2):
        counts = L.scan(conn, NOW, only=[t])
        assert (counts["updated"], counts["raised"], counts["reopened"], counts["errors"]) == (1, 0, 0, 0)
    assert len(notes(conn, owner)) == 1


@pytest.mark.parametrize("status", ["not_yet", "dismissed"])
def test_a_card_that_comes_back_tells_the_owner_again(world, status):
    conn, t, owner = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    _set_status(conn, t.key, status, 30)
    assert L.scan(conn, NOW, only=[t])["reopened"] == 0 and len(notes(conn, owner)) == 1, "still hidden: nobody is told"
    # 30 more: 63 decisions, two bands (510..800, 810..1100), to 1100 on 60 decisions: 60 >= 30 + 30
    fire(conn, t.key, [840 + 10 * i for i in range(30)], first=33)
    assert L.scan(conn, NOW, only=[t])["reopened"] == 1
    sid = _sid(conn, t.key)
    assert [(n[1], n[2]) for n in notes(conn, owner)] == [
        (f"Learning suggests a change to {t.key}: {PLAIN}", f"suggestion:{sid}"),
        (f"Learning suggests a change to {t.key}: Raise the limit on Refund amount from 500 USD to 1100 USD?",
         f"suggestion:{sid}")]


def test_a_superseded_card_raised_again_tells_the_owner_again(world):
    conn, t, owner = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    _set_status(conn, t.key, "superseded", None)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    assert len(notes(conn, owner)) == 2


@pytest.mark.parametrize("owner_value", ["", "   ", None])
def test_a_policy_with_no_owner_raises_its_card_and_tells_nobody(world, owner_value):
    conn, t, owner = world
    fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[replace(t, form={**t.form, "owner": owner_value})])["raised"] == 1
    assert by_change(conn, t.key)[CHANGE_500]["status"] == "open"
    # nothing to this test's owner, and nothing at all hanging on this policy's decisions
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_policy_notification n JOIN proc.bp_policy_firing f "
                "ON f.firing_id = n.firing_id WHERE f.policy_key = %s", (t.key,))
    assert notes(conn, owner) == [] and cur.fetchone()[0] == 0


def test_a_notification_that_cannot_be_written_does_not_lose_the_card(world, monkeypatch, caplog):
    conn, t, owner = world
    fire(conn, t.key, BASE)

    def broken(cur, firing_id, recipients, message, link):
        cur.execute("INSERT INTO proc.bp_policy_notification_no_such_table (recipient) VALUES (%s)", (owner,))

    monkeypatch.setattr(gate, "_notify", broken)
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[t])
    # a real failed statement inside the policy's transaction: without the savepoint the card is lost
    assert (counts["raised"], counts["errors"]) == (1, 0)
    [(status, confident, proposal, _stats, _n)] = row(conn, t.key)
    assert (status, confident, proposal["to"]) == ("open", True, 800)
    assert notes(conn, owner) == []
    assert any("could not be told" in r.getMessage() and t.key in r.getMessage() for r in caplog.records)
    assert conn.autocommit is True


def test_a_card_on_a_sensitive_field_is_never_raised_so_nobody_is_told(world):
    conn, t, owner = world
    fire(conn, t.key, BASE)
    marked = _marked_sensitive(t)
    assert L.scan(conn, NOW, only=[replace(marked, form={**marked.form, "owner": owner})])["raised"] == 0
    assert row(conn, t.key) == [] and notes(conn, owner) == []


def test_the_owner_reads_it_with_a_target_the_screen_can_open(world):
    conn, t, owner = world
    conn.cursor().execute(
        "INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) VALUES (%s, %s, 'test')",
        (owner, ["tst-b4-owner@example.test"]))
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    sid = _sid(conn, t.key)
    me = Principal(subject="tst-b4-owner", email="tst-b4-owner@example.test", claims={})
    [item] = [n for n in AV.my_notifications(conn, me, 50) if n["recipient"] == owner]
    assert item["target"] == {"kind": "suggestion", "id": sid, "policyKey": t.key}
    assert set(item) == {"id", "recipient", "message", "policyKey", "decisionId", "conflictId", "read", "createdAt",
                         "target"}
    assert (item["policyKey"], item["decisionId"], item["conflictId"], item["read"]) == (t.key, None, None, False)
    assert item["message"] == f"Learning suggests a change to {t.key}: {PLAIN}"
    # somebody who is not linked to the owner does not see it
    other = Principal(subject="tst-b4-other", email="tst-b4-other@example.test", claims={})
    assert [n for n in AV.my_notifications(conn, other, 50) if n["recipient"] == owner] == []
    # and reading it marks it read for the owner, as any notification
    assert AV.mark_read(conn, item["id"], me) == {"id": item["id"], "read": True}
