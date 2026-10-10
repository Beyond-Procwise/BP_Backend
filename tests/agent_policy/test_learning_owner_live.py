"""The owner's view of the queue, and who may hide a card, on bp_testdb (stage 5b, B3).
Needs PROCWISE_TEST_LIVE_DB=1.

test_learning_endpoints_live's world: a REAL live policy "TST …" owned by "TST Owner <tag>", with
one open suggestion; `tst-buyer` is linked to that owner in the decider map. Here `tst-buyer-2` is
a Buyer linked to nobody (or to another name), `tst-approver` an Approver linked to nobody.
Decider-map rows named "TST … <tag>" are removed by the world's teardown.
"""
import os

import pytest

from services.agent_policy import deciders
from services.agent_policy import learning as L
from tests.agent_policy import test_learning_endpoints_live as E
from tests.agent_policy.test_learning_endpoints_live import client, conn, world  # noqa: F401 - fixtures

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

SENTENCE = "Only the policy's owner or an approver can hide a suggestion."
OTHER_BUYER, ADMIN = E._hdr("tst-buyer-2"), E._hdr("tst-admin")


def _card_for(client, w, headers, query=""):
    r = client.get("/agent-policies/suggestions" + query, headers=headers)
    assert r.status_code == 200, r.text
    return next((c for c in r.json()["suggestions"] if c["policyKey"] == w.key), None)


def _flags(card):
    return card["ownerName"], card["mine"], card["canAct"]


def _link(conn, w, name, sub):
    conn.cursor().execute("INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) "
                          "VALUES (%s, %s, 'test')", (name, [f"{sub}@example.test"]))


def test_each_caller_sees_whether_the_card_is_theirs_and_whether_they_may_hide_it(client, conn, world):
    owner = f"TST Owner {world.tag}"
    assert _flags(_card_for(client, world, E.BUYER)) == (owner, True, True)
    assert _flags(_card_for(client, world, OTHER_BUYER)) == (owner, False, False)
    assert _flags(_card_for(client, world, E.APPROVER)) == (owner, False, True)
    assert _flags(_card_for(client, world, ADMIN)) == (owner, False, True)
    assert _flags(_card_for(client, world, E.VIEWER)) == (owner, False, False)
    r = client.get("/agent-policies/suggestions", headers=E.BUYER)
    assert "mineUnavailable" not in r.json()
    # the policy page's cards say the same, for the caller reading the page
    for headers, flags in ((E.BUYER, (owner, True, True)), (OTHER_BUYER, (owner, False, False)),
                           (E.APPROVER, (owner, False, True))):
        [card] = client.get(f"/agent-policies/{world.key}", headers=headers).json()["suggestions"]
        assert _flags(card) == flags


def test_mine_1_returns_the_owners_cards_only(client, conn, world):
    assert _card_for(client, world, E.BUYER, "?mine=1")["id"] == world.sid
    assert _card_for(client, world, OTHER_BUYER, "?mine=1") is None
    assert _card_for(client, world, E.APPROVER, "?mine=1") is None, "may hide it, but it is not theirs"
    assert _card_for(client, world, E.APPROVER)["id"] == world.sid
    r = client.get("/agent-policies/suggestions?mine=1", headers=E.BUYER)
    assert all(c["mine"] for c in r.json()["suggestions"])
    assert client.get("/agent-policies/suggestions?mine=0", headers=E.BUYER).status_code == 422


def test_the_owner_a_buyer_may_hide_the_card(client, conn, world):
    r = E._act(client, world, "not_yet", E.BUYER)
    assert r.status_code == 201, r.text
    assert (r.json()["status"], r.json()["mine"], r.json()["canAct"]) == ("not_yet", True, False)
    assert E._row(conn, world)[:3] == ("not_yet", 30, "tst-buyer")
    assert _flags(_card_for(client, world, E.BUYER, "?status=all")) == (f"TST Owner {world.tag}", True, False)


def test_a_buyer_who_is_not_the_owner_may_not_and_nothing_is_written(client, conn, world):
    del world.audits[:]
    for verb in ("not_yet", "dismiss"):
        r = E._act(client, world, verb, OTHER_BUYER)
        assert r.status_code == 403 and r.json() == {"detail": SENTENCE}, r.text
    assert E._row(conn, world) == ("open", None, None, None)
    assert [(a["action_type"], a["phase"], a["status"], a["details"].get("code")) for a in world.audits] == [
        ("agent_policy.suggestion_act", "authorize", "allowed", None),
        ("agent_policy.suggestion_act", "act", "refused", "not_allowed")] * 2
    assert conn.autocommit is True


@pytest.mark.parametrize("headers,who", [(E.APPROVER, "tst-approver"), (ADMIN, "tst-admin")])
def test_an_approver_or_above_may_hide_a_card_that_is_not_theirs(client, conn, world, headers, who):
    r = E._act(client, world, "dismiss", headers)
    assert r.status_code == 201, r.text
    assert (r.json()["status"], r.json()["mine"]) == ("dismissed", False)
    assert E._row(conn, world)[:3] == ("dismissed", 30, who)


def test_a_viewer_linked_to_the_owner_still_cannot(client, conn, world):
    conn.cursor().execute("UPDATE proc.bp_policy_decider_map SET emails = emails || %s WHERE decider_name = %s",
                          (["tst-viewer@example.test"], f"TST Owner {world.tag}"))
    assert _flags(_card_for(client, world, E.VIEWER)) == (f"TST Owner {world.tag}", True, False)
    assert E._act(client, world, "dismiss", E.VIEWER).status_code == 403
    assert E._row(conn, world)[0] == "open"


def test_draft_this_change_stays_with_any_buyer(client, conn, world):
    assert E._draft(client, world, OTHER_BUYER).status_code == 200
    assert E._draft(client, world, E.VIEWER).status_code == 403


def test_naming_yourself_owner_in_a_draft_does_not_make_the_card_yours(client, conn, world):
    mine = f"TST Mine {world.tag}"
    _link(conn, world, mine, "tst-buyer-2")
    r = E._save(client, world, E._live_form(conn, world, owner=mine), base=world.live, headers=OTHER_BUYER)
    assert r.status_code == 200, r.text
    assert _flags(_card_for(client, world, OTHER_BUYER)) == (f"TST Owner {world.tag}", False, False), \
        "the owner in force is the live version's"
    assert E._act(client, world, "dismiss", OTHER_BUYER).status_code == 403
    assert _flags(_card_for(client, world, E.BUYER)) == (f"TST Owner {world.tag}", True, True)
    assert E._act(client, world, "dismiss", E.BUYER).status_code == 201


def test_an_unreadable_decider_map_shows_everything_says_so_and_only_an_approver_may_hide(client, conn, world,
                                                                                          monkeypatch):
    def boom(conn):
        raise RuntimeError("TST map down")
    monkeypatch.setattr(deciders, "load_map", boom)
    for query in ("", "?mine=1"):
        r = client.get("/agent-policies/suggestions" + query, headers=E.BUYER)
        assert r.status_code == 200 and r.json()["mineUnavailable"] is True, r.text
        [card] = [c for c in r.json()["suggestions"] if c["policyKey"] == world.key]
        assert _flags(card) == (f"TST Owner {world.tag}", False, False)
        assert not any(c["mine"] for c in r.json()["suggestions"])
    assert E._act(client, world, "dismiss", E.BUYER).status_code == 403, "the owner cannot be told apart"
    assert E._row(conn, world)[0] == "open"
    assert _flags(_card_for(client, world, E.APPROVER)) == (f"TST Owner {world.tag}", False, True)
    assert E._act(client, world, "dismiss", E.APPROVER).status_code == 201


def test_the_rule_is_the_service_functions_own(conn, world):
    """Whatever the endpoint checked: act() judges the caller it is given, under the lock."""
    with pytest.raises(L.Refused) as refused:
        L.act(conn, world.sid, "dismiss", actor="x", now=E.NOW, caller=L.Caller(editor=True))
    assert (refused.value.code, refused.value.message) == ("not_allowed", SENTENCE)
    with pytest.raises(L.Refused) as refused:
        L.act(conn, world.sid, "dismiss", actor="x", now=E.NOW,
              caller=L.Caller(names=frozenset({f"TST Owner {world.tag}"})))
    assert refused.value.code == "not_allowed", "linked, but below the role that edits drafts"
    assert E._row(conn, world)[0] == "open" and conn.autocommit is True
    with pytest.raises(TypeError):
        L.act(conn, world.sid, "dismiss", actor="x", now=E.NOW)      # the caller is never optional
    out = L.act(conn, world.sid, "dismiss", actor="x", now=E.NOW,
                caller=L.Caller(names=frozenset({f"TST Owner {world.tag}"}), editor=True))
    assert (out["status"], out["mine"], out["canAct"]) == ("dismissed", True, False)
