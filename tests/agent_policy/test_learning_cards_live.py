"""The suggestion card, the queue and a person's actions on a card.

The `card` tests are pure. The rest need PROCWISE_TEST_LIVE_DB=1 and reuse the scan tests' world:
keys TST-<hex>a that are not real policies, so a card has no policy form to read and falls back
to the wording stored in the proposal. Cards of a REAL policy are in test_learning_endpoints_live.

The numbers (see test_learning_scan_live's docstring for the banding rule): the base data is 33
approvals at 510, 520, ... 830 under "args.amount gt 500". Band 1 = 510..800 passes, the 3 above
it are not a band -> the suggestion is 500 -> 800 on 30 decisions by 3 people over 40 days.
"""
import copy
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning as L
from tests.agent_policy.fixtures import FORM_EXAMPLE
from tests.agent_policy.test_learning_scan_live import BASE, NOW, fire, row, world  # noqa: F401 - world is a fixture

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
LATER = NOW + timedelta(hours=2)
#: an Approver: may hide any card (who may is test_learning_owner's and test_learning_owner_live's)
ANYONE = L.Caller(editor=True, approver=True)
EVIDENCE_30 = "30 of 30 requests between 500 USD and 800 USD were approved, by 3 different people over 40 days."


def _proposal():
    return {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 800,
            "changeKey": "threshold:args.amount:gt:500", "stats": _stats(), "bands": 1,
            "plain": "STORED plain", "evidence": "STORED evidence", "inputName": "Refund amount"}


def _stats(decisions=30, approved=30, **over):
    return {"decisions": decisions, "approved": approved, "wilsonLower": 0.8865, "approvers": 3, "days": 40,
            "edited": 0, "reversed": 0, "medianSeconds": 60.0, **over}


def _row(**over):
    proposal = _proposal()
    base = {"suggestion_id": 12, "policy_key": "FIN-0012", "status": "open", "kind": "threshold",
            "proposal": proposal, "stats": _stats(), "confident": True,
            "updated_at": datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc), "form": copy.deepcopy(FORM_EXAMPLE)}
    base.update(over)
    return base


GOOD = _proposal()


# ------------------------------------------------------------------ card (pure)
def test_a_card_is_built_from_the_policy_form_and_the_stats_column():
    c = L.card(_row())
    assert c == {"id": 12, "policyKey": "FIN-0012", "policyName": "Refund or credit over $500", "status": "open",
                 "kind": "threshold", "change": "Raise the limit on Refund amount from 500 USD to 800 USD?",
                 "evidence": EVIDENCE_30,
                 "stats": {"decisions": 30, "approved": 30, "approvers": 3, "days": 40, "medianSeconds": 60.0,
                           "wilsonLower": 0.8865, "reversed": 0},
                 "suggestionConfidence": "High", "canDraft": True, "updatedAt": "2026-10-10T09:00:00+00:00",
                 # stage 5b: whose card, and may the caller hide it (nobody said who is asking)
                 "ownerName": "", "mine": False, "canAct": False}
    assert "confidence" not in c, "extraction confidence and suggestion confidence never share a name"


def test_a_weak_card_shows_the_evidence_as_it_now_stands_not_as_it_was_when_last_supported():
    # the stored proposal still says 30 of 30; the stats column says one of 31 was rejected since
    c = L.card(_row(confident=False, stats=_stats(31, 30, wilsonLower=0.8381)))
    assert c["evidence"] == ("30 of 31 requests between 500 USD and 800 USD were approved, "
                             "by 3 different people over 40 days.")
    assert c["stats"]["decisions"] == 31 and c["stats"]["wilsonLower"] == 0.8381
    assert c["suggestionConfidence"] == "Not enough yet" and c["canDraft"] is False


@pytest.mark.parametrize("reversed_,said", [(1, " 1 of them was later marked as reversed."),
                                            (2, " 2 of them were later marked as reversed."),
                                            (12, " 12 of them were later marked as reversed.")])
def test_a_card_weakened_by_a_reversal_says_so(reversed_, said):
    # 30 of 30 approved reads as full support: without the sentence nothing says why it is "Not enough yet"
    c = L.card(_row(confident=False, stats=_stats(reversed=reversed_)))
    assert c["evidence"] == EVIDENCE_30 + said
    assert c["stats"]["reversed"] == reversed_ and c["suggestionConfidence"] == "Not enough yet"
    gone = L.card(_row(confident=False, stats=_stats(reversed=reversed_), form=None))
    assert gone["evidence"] == "STORED evidence" + said, "also on the wording stored with the card"


@pytest.mark.parametrize("stats", [_stats(), _stats(reversed=None), _stats(reversed=True), _stats(reversed="2"),
                                   {k: v for k, v in _stats().items() if k != "reversed"}])
def test_a_card_with_no_reversal_says_nothing_about_one(stats):
    c = L.card(_row(stats=stats))
    assert c["evidence"] == EVIDENCE_30 and "reversed" in c["stats"]


def test_a_card_whose_policy_is_gone_uses_the_wording_stored_with_it():
    c = L.card(_row(form=None))
    assert (c["policyName"], c["change"], c["evidence"]) == ("", "STORED plain", "STORED evidence")


@pytest.mark.parametrize("status", ["not_yet", "dismissed", "saved", "superseded"])
def test_only_an_open_confident_card_can_be_drafted(status):
    c = L.card(_row(status=status))
    assert c["status"] == status and c["canDraft"] is False and c["suggestionConfidence"] == "High"


def test_a_list_card_reads_as_a_list_change():
    proposal = {"kind": "list", "field": "tool.name", "op": "in", "remove": "credit.issue",
                "changeKey": "list:tool.name:credit.issue", "stats": _stats()}
    c = L.card(_row(kind="list", proposal=proposal))
    assert c["kind"] == "list" and c["change"] == "Stop asking for approval when Tool is credit.issue?"
    assert c["evidence"] == ("30 of 30 requests where Tool is credit.issue were approved, "
                             "by 3 different people over 40 days.")


def test_a_card_never_fails_on_a_proposal_it_cannot_read(caplog):
    c = L.card(_row(proposal={"kind": "threshold"}, stats={}))
    assert (c["change"], c["evidence"], c["stats"]["decisions"]) == ("", "", None)
    [said] = [r.getMessage() for r in caplog.records]
    assert "suggestion 12" in said and "KeyError" in said, "enough to find the row and the reason"
    assert L.card(_row(proposal=None, form=None))["change"] == ""


# ------------------------------------------------------------------ against the database
def _sid(conn, key):
    return L.for_policy(conn.cursor(), key)[0]["id"]


def _full(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT status, n_at_action, acted_by, acted_at, saved_version FROM proc.bp_agent_policy_suggestion "
                "WHERE policy_key = %s ORDER BY suggestion_id", (key,))
    return cur.fetchall()


def _limit_form(value):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["hidden"]["condition"]["all"][1]["value"] = value
    return form


@live
def test_the_card_names_suggestion_confidence_and_never_extraction_confidence(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    [c] = [c for c in L.list_cards(conn) if c["policyKey"] == t.key]
    assert c["suggestionConfidence"] == "High" and c["canDraft"] is True and "confidence" not in c
    assert c["change"].startswith("Raise the limit") and c["stats"]["approvers"] == 3
    # band 1 = 510..800: 30 decisions, all approved; no policy row, so the stored sentence
    assert (c["stats"]["decisions"], c["stats"]["approved"], c["policyName"]) == (30, 30, "")
    assert c["change"] == "Raise the limit on Refund amount from 500 USD to 800 USD?" and c["evidence"] == EVIDENCE_30
    assert L.for_policy(conn.cursor(), t.key) == [c], "the policy page and the queue show the same card"


@live
def test_not_yet_and_dismiss_record_who_when_and_how_many(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    sid = _sid(conn, t.key)
    out = L.act(conn, sid, "not_yet", actor="u1", now=LATER, caller=ANYONE)
    assert (out["id"], out["status"], out["canDraft"]) == (sid, "not_yet", False)
    assert L.for_policy(conn.cursor(), t.key) == [], "hidden from the form"
    assert row(conn, t.key)[0][4] == 30, "the 30 decisions of band 1 (510..800), from the stats column"
    assert _full(conn, t.key) == [("not_yet", 30, "u1", LATER, None)]
    assert conn.autocommit is True, "the connection is handed back as it came"
    with pytest.raises(L.Refused) as exc:
        L.act(conn, sid, "dismiss", actor="u2", now=LATER, caller=ANYONE)
    assert (exc.value.code, exc.value.message) == ("not_open", "This suggestion is no longer open.")
    assert _full(conn, t.key) == [("not_yet", 30, "u1", LATER, None)], "a refused action writes nothing"
    # the scan keeps it hidden: 33 decisions, 30 released, and 30 < 30 + not_yet_more (30)
    counts = L.scan(conn, NOW, only=[t])
    assert counts["reopened"] == 0 and row(conn, t.key)[0][0] == "not_yet"


@live
def test_dismiss_closes_the_card_and_the_queue_shows_it_only_under_all(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    sid = _sid(conn, t.key)
    assert L.act(conn, sid, "dismiss", actor="u1", now=LATER, caller=ANYONE)["status"] == "dismissed"
    assert _full(conn, t.key) == [("dismissed", 30, "u1", LATER, None)]
    assert [c for c in L.list_cards(conn) if c["policyKey"] == t.key] == []
    [c] = [c for c in L.list_cards(conn, status="all") if c["policyKey"] == t.key]
    assert (c["id"], c["status"], c["canDraft"]) == (sid, "dismissed", False)


@live
def test_the_count_recorded_on_a_weak_card_is_the_evidence_as_it_now_stands(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    # A rejection at 605 falls inside band 1, which is now the 30 nearest the limit: 510..600 (10),
    # 605, 610..790 (19) = 29 approved of 30, Wilson lower 0.8333 < 0.85. Nothing is proposed and the
    # card turns weak. The released set of the stored change, (500, 800], is 31 decisions, 30 approved.
    fire(conn, t.key, [605], result="rejected", first=33)
    assert L.scan(conn, NOW, only=[t])["weakened"] == 1
    [c] = L.for_policy(conn.cursor(), t.key)
    assert (c["suggestionConfidence"], c["canDraft"]) == ("Not enough yet", False)
    assert (c["stats"]["decisions"], c["stats"]["approved"]) == (31, 30)
    L.act(conn, c["id"], "not_yet", actor="u1", now=LATER, caller=ANYONE)
    assert _full(conn, t.key)[0][:2] == ("not_yet", 31), "31 from the stats column, not the proposal's stale 30"


@live
def test_a_weak_card_cannot_be_drafted(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET confident = false WHERE policy_key = %s",
                          (t.key,))
    sid = _sid(conn, t.key)
    assert L.for_policy(conn.cursor(), t.key)[0]["suggestionConfidence"] == "Not enough yet"
    with pytest.raises(L.Refused) as exc:
        L.draft(conn, sid)
    assert (exc.value.code, exc.value.message) == ("not_confident", "There is not enough evidence yet.")


@live
def test_a_card_whose_policy_is_gone_cannot_be_drafted(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    with pytest.raises(L.Refused) as exc:
        L.draft(conn, _sid(conn, t.key))          # TST-… is not a policy row
    assert exc.value.code == "policy_changed"


@live
def test_a_hidden_card_cannot_be_drafted(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    sid = _sid(conn, t.key)
    L.act(conn, sid, "not_yet", actor="u1", now=LATER, caller=ANYONE)
    with pytest.raises(L.Refused) as exc:
        L.draft(conn, sid)
    assert exc.value.code == "not_open"


@live
def test_an_unknown_suggestion_and_an_unknown_verb(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    for call in (lambda: L.act(conn, -1, "dismiss", actor="u1", now=LATER, caller=ANYONE),
                 lambda: L.draft(conn, -1)):
        with pytest.raises(L.Refused) as exc:
            call()
        assert exc.value.code == "not_found"
    with pytest.raises(ValueError):
        L.act(conn, _sid(conn, t.key), "apply", actor="u1", now=LATER, caller=ANYONE)
    assert _full(conn, t.key) == [("open", None, None, None, None)]


@live
@pytest.mark.parametrize("limit,saved", [(500, 0), (400, 0), (800, 1), (600, 1)])
def test_a_save_marks_the_suggestion_saved_only_when_the_saved_form_holds_the_change(world, limit, saved):
    # 500: unchanged. 400: tightened, the other way. 800: the suggested change. 600: looser than 500
    # in the suggested direction, typed by hand.
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    assert L.note_save(conn.cursor(), t.key, 7, _limit_form(limit)) == saved
    assert _full(conn, t.key)[0][0::4] == (("saved", 7) if saved else ("open", None))
    if saved:
        assert L.for_policy(conn.cursor(), t.key) == []
        assert L.note_save(conn.cursor(), t.key, 8, _limit_form(limit)) == 0, "saved once"
        assert _full(conn, t.key)[0][4] == 7


@live
@pytest.mark.parametrize("verb,status,saved", [("not_yet", "not_yet", 1), ("dismiss", "dismissed", 0)])
def test_a_snoozed_card_is_marked_saved_and_a_dismissed_one_is_left(world, verb, status, saved):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    L.act(conn, _sid(conn, t.key), verb, actor="u1", now=LATER, caller=ANYONE)
    assert L.note_save(conn.cursor(), t.key, 3, _limit_form(800)) == saved
    assert _full(conn, t.key)[0][0] == ("saved" if saved else status)


@live
def test_a_save_of_another_policy_or_an_unreadable_form_marks_nothing(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    assert L.note_save(conn.cursor(), t.key[:-1] + "b", 2, _limit_form(800)) == 0
    for form in ({}, {"hidden": None}, {"hidden": {"condition": "nonsense"}}):
        assert L.note_save(conn.cursor(), t.key, 2, form) == 0
    assert _full(conn, t.key)[0][0] == "open"


@live
def test_one_unreadable_proposal_does_not_block_the_policys_other_suggestions(world, caplog):
    conn, t = world
    cur = conn.cursor()
    # two rows of the same policy whose proposal cannot be read (no field, no op), one older and
    # one newer than the real suggestion, so the real one is reached whatever the order
    ids = []
    for change in ("broken-before", "broken-after"):
        cur.execute("INSERT INTO proc.bp_agent_policy_suggestion (policy_key, change_key, policy_version, kind, "
                    "proposal, stats, confident, status) VALUES (%s, %s, 1, 'threshold', %s, '{}', true, 'open') "
                    "RETURNING suggestion_id", (t.key, change, '{"kind": "threshold"}'))
        ids.append(cur.fetchone()[0])
        if change == "broken-before":
            cur.execute("INSERT INTO proc.bp_agent_policy_suggestion (policy_key, change_key, policy_version, kind, "
                        "proposal, stats, confident, status) VALUES (%s, %s, 1, 'threshold', %s, %s, true, 'open')",
                        (t.key, "threshold:args.amount:gt:500", json.dumps(GOOD), json.dumps(GOOD["stats"])))
    assert L.note_save(conn.cursor(), t.key, 7, _limit_form(800)) == 1
    assert [(r[0], r[4]) for r in _full(conn, t.key)] == [("open", None), ("saved", 7), ("open", None)]
    said = [r.getMessage() for r in caplog.records]
    for broken in ids:
        assert [m for m in said if f"suggestion {broken} " in m and "KeyError" in m], "logged by suggestion id"
