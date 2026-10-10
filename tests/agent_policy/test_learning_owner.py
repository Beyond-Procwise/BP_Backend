"""Whose card is whose, and who may hide one (stage 5b, B3): the pure rule, and the endpoints
through the WHOLE app with no database. The real rows are test_learning_owner_live's.

The rule: Not yet / Dismiss need the role that edits drafts (Buyer) AND either a link to the
policy's owner in the decider map or the Approver role. `mine` is the link; `canAct` is the rule.
"""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import deciders
from services.agent_policy import learning as L
from tests.agent_policy.test_learning_cards_live import _row
from tests.agent_policy.test_learning_endpoints_scrub import _card, _FakeConnCtx

SENTENCE = "Only the policy's owner or an approver can hide a suggestion."
OWNER = "Finance Manager"
HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@example.test", "X-User-Groups": json.dumps([])}

BUYER_OWNER = L.Caller(names=frozenset({OWNER}), editor=True)
BUYER = L.Caller(names=frozenset({"Someone Else"}), editor=True)
APPROVER = L.Caller(editor=True, approver=True)
VIEWER_OWNER = L.Caller(names=frozenset({OWNER}))


# ------------------------------------------------------------------ the card (pure)
@pytest.mark.parametrize("caller,mine,can_act", [
    (BUYER_OWNER, True, True),
    (BUYER, False, False),
    (APPROVER, False, True),
    (L.Caller(names=frozenset({OWNER}), editor=True, approver=True), True, True),
    (VIEWER_OWNER, True, False),             # linked, but below the role that may act at all
    (L.Caller(), False, False),
    (None, False, False),                    # nobody said who is asking
    (L.Caller(editor=True, approver=True, known=False), False, True),    # the map could not be read
    (L.Caller(editor=True, known=False), False, False),
])
def test_a_card_says_whether_it_is_the_callers_and_whether_they_may_hide_it(caller, mine, can_act):
    c = L.card(_row(owner=OWNER), caller)
    assert (c["ownerName"], c["mine"], c["canAct"]) == (OWNER, mine, can_act)
    assert type(c["mine"]) is bool and type(c["canAct"]) is bool


@pytest.mark.parametrize("status", ["not_yet", "dismissed", "saved", "superseded"])
def test_a_card_that_is_not_open_cannot_be_acted_on_by_anyone(status):
    c = L.card(_row(owner=OWNER, status=status), L.Caller(names=frozenset({OWNER}), editor=True, approver=True))
    assert (c["mine"], c["canAct"]) == (True, False)


@pytest.mark.parametrize("owner", [None, "", "   "])
def test_a_policy_with_no_owner_in_force_is_nobodys(owner):
    linked_to_blank = L.Caller(names=frozenset({"", "   "}), editor=True)
    c = L.card(_row(owner=owner), linked_to_blank)
    assert (c["ownerName"], c["mine"], c["canAct"]) == ("", False, False)
    assert L.card(_row(owner=owner), APPROVER)["canAct"] is True


def test_the_owner_is_the_one_in_force_not_the_one_the_latest_form_names():
    row = _row(owner=OWNER)
    row["form"]["owner"] = "Someone Else"            # a newer draft renamed the owner
    c = L.card(row, BUYER)
    assert (c["ownerName"], c["mine"], c["canAct"]) == (OWNER, False, False)


def test_the_cards_other_keys_are_what_they_were():
    before = {"id", "policyKey", "policyName", "status", "kind", "change", "evidence", "stats",
              "suggestionConfidence", "canDraft", "updatedAt"}
    plain, owned = L.card(_row(owner=OWNER)), L.card(_row(owner=OWNER), BUYER_OWNER)
    assert set(plain) == before | {"mine", "ownerName", "canAct"}
    assert {k: plain[k] for k in before} == {k: owned[k] for k in before}, "who asks changes only the new keys"


def test_who_may_hide():
    assert BUYER_OWNER.may_hide(OWNER) and APPROVER.may_hide(OWNER) and APPROVER.may_hide("")
    assert not BUYER.may_hide(OWNER) and not VIEWER_OWNER.may_hide(OWNER) and not BUYER_OWNER.may_hide("")
    assert not L.Caller(approver=True).may_hide(OWNER), "never below the role that edits drafts"


def test_the_caller_is_read_from_the_decider_map_and_an_unreadable_map_is_said_not_raised(monkeypatch, caplog):
    who = R.Principal(subject="u1", email="U1@Example.test", claims={"cognito:groups": ["G1"]})
    monkeypatch.setattr(deciders, "load_map", lambda conn: {
        "By Email": {"groups": [], "emails": ["u1@example.test"]}, "By Group": {"groups": ["G1"], "emails": []},
        "Not Mine": {"groups": ["G2"], "emails": ["x@example.test"]}, "Nobody": {"groups": [], "emails": []}})
    got = L.caller_of(object(), who, editor=True, approver=False)
    assert got == L.Caller(names=frozenset({"By Email", "By Group"}), editor=True, approver=False, known=True)

    def boom(conn):
        raise RuntimeError("TST map down")
    monkeypatch.setattr(deciders, "load_map", boom)
    with caplog.at_level(logging.WARNING, logger="services.agent_policy.learning"):
        got = L.caller_of(object(), who, editor=True, approver=True)
    assert got == L.Caller(names=frozenset(), editor=True, approver=True, known=False)
    assert any("decider map" in r.getMessage() for r in caplog.records)
    # a map the endpoint already read is used as it is: the (failing) read is not made again
    got = L.caller_of(object(), who, editor=True, approver=False,
                      mapping={"By Group": {"groups": ["G1"], "emails": []}})
    assert got == L.Caller(names=frozenset({"By Group"}), editor=True)
    assert L.caller_of(object(), who, editor=False, approver=False, mapping={}) == L.Caller()


# ------------------------------------------------------------------ through the whole app (no database)
@pytest.fixture
def state():
    return {"role": "Buyer", "audits": [], "map": {OWNER: {"groups": [], "emails": ["u1@example.test"]}},
            "cards": [_card(id=1, ownerName=OWNER), _card(id=2, ownerName="Someone Else")], "seen": {}}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: state["role"])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: state["audits"].append(kw))

    def load_map(conn):
        if state["map"] is None:
            raise RuntimeError("TST map down")
        return state["map"]

    def list_cards(conn, *, status="open", caller=None):
        state["seen"]["caller"] = caller
        return [{**c, "mine": c["ownerName"] in caller.names, "canAct": caller.may_hide(c["ownerName"])}
                for c in state["cards"]]

    def act(conn, sid, verb, *, actor, now, caller):
        state["seen"]["caller"] = caller
        owner = next(c["ownerName"] for c in state["cards"] if c["id"] == sid)
        if not caller.may_hide(owner):
            raise L.Refused(*L.NOT_ALLOWED)
        return {**_card(id=sid, status="dismissed", canDraft=False), "mine": owner in caller.names, "canAct": False}

    monkeypatch.setattr(deciders, "load_map", load_map)
    monkeypatch.setattr(L, "list_cards", list_cards)
    monkeypatch.setattr(L, "act", act)
    return TestClient(app)


def _ids(r):
    assert r.status_code == 200, r.text
    return [c["id"] for c in r.json()["suggestions"]]


@pytest.mark.parametrize("role,editor,approver", [("Viewer", False, False), ("Buyer", True, False),
                                                  ("Approver", True, True), ("Admin", True, True)])
def test_the_queue_is_read_as_the_caller(client, state, role, editor, approver):
    state["role"] = role
    r = client.get("/agent-policies/suggestions", headers=HDR)
    assert _ids(r) == [1, 2] and "mineUnavailable" not in r.json()
    assert state["seen"]["caller"] == L.Caller(names=frozenset({OWNER}), editor=editor, approver=approver)
    assert [(c["mine"], c["canAct"]) for c in r.json()["suggestions"]] == \
        [(True, editor), (False, approver)], "readable after the output filter"


def test_mine_1_returns_only_the_callers_cards(client, state):
    assert _ids(client.get("/agent-policies/suggestions?mine=1", headers=HDR)) == [1]
    assert _ids(client.get("/agent-policies/suggestions?mine=1&status=all", headers=HDR)) == [1]
    state["map"] = {}
    assert _ids(client.get("/agent-policies/suggestions?mine=1", headers=HDR)) == []
    for bad in ("0", "true", "yes", "", "11"):
        assert client.get(f"/agent-policies/suggestions?mine={bad}", headers=HDR).status_code == 422, bad


def test_an_unreadable_decider_map_is_said_and_is_never_an_error(client, state):
    state["map"] = None
    for query in ("", "?mine=1"):                    # nothing can be called "mine": everything is shown
        r = client.get("/agent-policies/suggestions" + query, headers=HDR)
        assert _ids(r) == [1, 2] and r.json()["mineUnavailable"] is True
        assert [(c["mine"], c["canAct"]) for c in r.json()["suggestions"]] == [(False, False)] * 2
    state["role"] = "Approver"
    r = client.get("/agent-policies/suggestions", headers=HDR)
    assert [(c["mine"], c["canAct"]) for c in r.json()["suggestions"]] == [(False, True)] * 2
    assert r.json()["mineUnavailable"] is True


def test_a_buyer_who_is_not_the_owner_is_refused_in_these_words_and_it_is_audited(client, state):
    r = client.post("/agent-policies/suggestions/2/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 403 and r.json() == {"detail": SENTENCE}, "the sentence arrives after the output filter"
    assert [(a["action_type"], a["phase"], a["status"]) for a in state["audits"]] == [
        ("agent_policy.suggestion_act", "authorize", "allowed"), ("agent_policy.suggestion_act", "act", "refused")]
    assert state["audits"][1]["details"]["code"] == "not_allowed"
    assert state["audits"][0]["details"]["minimum"] == "Buyer", "the floor is still Buyer"


def test_the_owner_and_an_approver_may_hide_and_a_viewer_never(client, state):
    assert client.post("/agent-policies/suggestions/1/act", json={"verb": "not_yet"}, headers=HDR).status_code == 201
    state["role"] = "Approver"
    assert client.post("/agent-policies/suggestions/2/act", json={"verb": "dismiss"}, headers=HDR).status_code == 201
    state["role"] = "Viewer"                         # linked to the owner of card 1, but below the floor
    r = client.post("/agent-policies/suggestions/1/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 403 and r.json()["detail"] != SENTENCE, "refused by the role floor, before the rule"


def test_hiding_with_the_decider_map_unreadable(client, state):
    state["map"] = None
    r = client.post("/agent-policies/suggestions/1/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 403 and r.json() == {"detail": SENTENCE}, "the owner cannot be told apart: not allowed"
    state["role"] = "Approver"
    assert client.post("/agent-policies/suggestions/1/act", json={"verb": "dismiss"}, headers=HDR).status_code == 201
