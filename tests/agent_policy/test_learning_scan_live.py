"""The learning scan against the real firing log. Needs PROCWISE_TEST_LIVE_DB=1.

Targets are injected (keys TST-<hex>a, never real policies). Firing rows are append-only and
stay; suggestion rows are removed by the fixture.

How every expected number here is worked (the banding rule, learning_patterns._threshold). The
policy is "args.amount gt 500". Decisions above 500 are ordered by amount and cut into bands of
min_decisions (30) walking away from the limit; each band must pass every guardrail alone; a
tail shorter than 30 is not evidence. `fire` gives decision i the amount it is told, asks it on
slot (7 * i) % 30 of 30 slots spread over 40 days and has person i % 3 decide it 60 s later.
7 and 30 share no factor, so ANY 30 consecutive i cover all 30 slots (first to last = 40 days)
and all 3 people: a band of 30 neighbouring amounts passes days (40 >= 30), approvers (3 >= 3)
and median time (60 s >= 30 s) on its own. 30 approved of 30 has a Wilson lower bound of
30 / (30 + 1.96^2) = 0.8865 >= 0.85.

The base data, used by most tests: 33 approvals, i = 0..32, amounts 510, 520, ... 830.
  band 1 = i 0..29 = 510..800, passes. The 3 left (810, 820, 830) are not a band.
  -> to = 800 (already two significant figures), released set (500, 800] = 30 decisions.
"""
import copy
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning as L
from services.agent_policy import learning_patterns as LP
from services.agent_policy import settings as S
from services.agent_policy.enforcement import MASK
from services.db import get_conn
from tests.agent_policy.fixtures import FORM_EXAMPLE

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)
START = NOW - timedelta(days=45)
BASE = [510 + 10 * i for i in range(33)]
CHANGE_500 = "threshold:args.amount:gt:500"
CFG = S.DEFAULTS["learning"]


def _siblings(key):
    """The keys one test may use: its own and two more for tests that scan several policies."""
    return [key, key[:-1] + "b", key[:-1] + "c"]


@pytest.fixture
def world():
    key = f"TST-{uuid.uuid4().hex[:6]}a"
    form = copy.deepcopy(FORM_EXAMPLE)
    with get_conn() as conn:
        yield conn, L.Target(key=key, live_version=1, form=form, versions=[1])
        conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = ANY(%s)",
                              (_siblings(key),))


def fire(conn, key, amounts, *, result="approved", seconds=60, version=1, decision_ids=None, by=None,
         first=0, at=None, extra=None, asked_at=None):
    """One settled approve firing per amount. Row number p is decision i = at[p] (or first + p):
    asked on slot (7 * i) % 30 of 30 slots over 40 days, decided by person i % 3. Time and decider
    do not follow the amount (see the module docstring)."""
    ids = []
    cur = conn.cursor()
    for p, amount in enumerate(amounts):
        i = at[p] if at is not None else first + p
        asked = asked_at or START + timedelta(days=40 * ((7 * i) % 30) / 29)
        values = extra[p] if extra else {}
        cur.execute(
            "INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, outcome, "
            "result, matched_values, decision_id, decided_level, decided_by, decided_at, created_at) "
            "VALUES (%s,%s,'tool.call.before','tst','approve',%s,%s,%s,0,%s,%s,%s) RETURNING firing_id",
            (key, version, result, json.dumps({"tool.name": "refund.issue", "args.amount": amount, **values}),
             decision_ids[p] if decision_ids else None, by or f"person-{i % 3}",
             asked + timedelta(seconds=seconds), asked))
        ids.append(cur.fetchone()[0])
    return ids


def row(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT status, confident, proposal, stats, n_at_action FROM proc.bp_agent_policy_suggestion "
                "WHERE policy_key = %s ORDER BY suggestion_id", (key,))
    return cur.fetchall()


def by_change(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT change_key, status, confident, proposal, stats, policy_version "
                "FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (key,))
    return {r[0]: {"status": r[1], "confident": r[2], "proposal": r[3], "stats": r[4], "version": r[5]}
            for r in cur.fetchall()}


def _case_ids(n):
    """n approval-case ids nobody else uses (the column has no foreign key)."""
    base = 10 ** 15 + uuid.uuid4().int % 10 ** 15
    return [base + j for j in range(n)]


def _reverse(conn, firing_id):
    conn.cursor().execute("INSERT INTO proc.bp_policy_firing_reversal (firing_id, reversed_by, reason) "
                          "VALUES (%s, 'tst', 'TST clawed back')", (firing_id,))


def _set_status(conn, key, status, n):
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = %s, n_at_action = %s "
                          "WHERE policy_key = %s", (status, n, key))


def _limit(target, value):
    form = copy.deepcopy(target.form)
    form["hidden"]["condition"]["all"][1]["value"] = value
    return L.Target(key=target.key, live_version=target.live_version, form=form, versions=target.versions)


def test_the_helper_spreads_any_thirty_neighbours_over_forty_days_and_three_people(world):
    conn, t = world
    fire(conn, t.key, BASE)
    decided = sorted(L.decisions(conn.cursor(), t.key, [1]), key=lambda d: d.values["args.amount"])
    for start in (0, 3):
        band = decided[start:start + 30]
        when = [d.decided_at for d in band]
        assert ((max(when) - min(when)).days, len({d.decided_by for d in band})) == (40, 3)


def test_thirty_approvals_by_three_people_over_thirty_days_raise_one_suggestion(world):
    conn, t = world
    fire(conn, t.key, BASE)
    counts = L.scan(conn, NOW, only=[t])
    assert counts["raised"] == 1 and counts["errors"] == 0 and counts["policies"] == 1
    [(status, confident, proposal, stats, _)] = row(conn, t.key)
    # band 1 = 510..800 passes; 810, 820, 830 are a tail of 3, not a band -> to 800, 30 released
    assert (status, confident, proposal["from"], proposal["to"]) == ("open", True, 500, 800), \
        "one band of 30 (510..800); the 3 above it are not a band"
    assert proposal["bands"] == 1 and proposal["inputName"] == "Refund amount"
    assert proposal["plain"] == "Raise the limit on Refund amount from 500 USD to 800 USD?"
    assert proposal["evidence"] == ("30 of 30 requests between 500 USD and 800 USD were approved, "
                                    "by 3 different people over 40 days.")
    assert (stats["decisions"], stats["approved"], stats["approvers"], stats["days"]) == (30, 30, 3, 40)
    assert by_change(conn, t.key)[CHANGE_500]["version"] == 1
    again = L.scan(conn, NOW, only=[t])
    assert again["updated"] == 1 and again["raised"] == 0 and len(row(conn, t.key)) == 1, \
        "a second scan adds nothing"


def test_one_decision_on_several_linked_rows_counts_once(world):
    conn, t = world
    cases = _case_ids(6)
    # 30 rows at 510..800 with full spread: counted as rows they are exactly band 1 and WOULD raise.
    # They are 6 approval cases, 5 linked rows each: 6 decisions, fewer than a band of 30.
    fire(conn, t.key, BASE[:30], decision_ids=[cases[p % 6] for p in range(30)])
    counts = L.scan(conn, NOW, only=[t])
    assert row(conn, t.key) == [] and counts["raised"] == 0 and counts["errors"] == 0, \
        "6 decisions, however many rows they were written on"
    assert len(L.decisions(conn.cursor(), t.key, [1])) == 6


def test_timeouts_and_system_decisions_are_not_evidence(world):
    conn, t = world
    # each group alone is 30 at 510..800 with full spread: band 1, if it counted
    fire(conn, t.key, BASE[:30], result="timed_out")
    fire(conn, t.key, BASE[:30], by="system:precedent")
    fire(conn, t.key, BASE[:30], by="system:group")
    fire(conn, t.key, BASE[:30], by="  ")
    fire(conn, t.key, BASE[:30], by="System:timeout")
    fire(conn, t.key, BASE[:30], by=" system:group")
    counts = L.scan(conn, NOW, only=[t])
    assert L.decisions(conn.cursor(), t.key, [1]) == []
    assert row(conn, t.key) == [] and counts["errors"] == 0


def test_a_masked_value_is_not_evidence(world):
    conn, t = world
    fire(conn, t.key, [MASK] * 40)
    counts = L.scan(conn, NOW, only=[t])
    assert len(L.decisions(conn.cursor(), t.key, [1])) == 40, "they are decisions, with no readable amount"
    assert row(conn, t.key) == [] and counts["errors"] == 0 and counts["policies"] == 1


def test_a_reversal_blocks_the_band_and_shows_not_enough_yet(world):
    conn, t = world
    ids = fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    _reverse(conn, ids[BASE.index(600)])
    # 600 is in band 1 (510..800): the band now holds a reversal and fails, so nothing is proposed.
    # The card stays, about the same change, with the evidence as it now stands: the released set
    # (500, 800] is still 30 decisions, 30 approved, 1 of them reversed.
    counts = L.scan(conn, NOW, only=[t])
    assert counts["weakened"] == 1 and counts["updated"] == 0 and counts["errors"] == 0
    [(status, confident, proposal, stats, _)] = row(conn, t.key)
    assert (status, confident, proposal["to"]) == ("open", False, 800)
    assert (stats["reversed"], stats["decisions"], stats["approved"]) == (1, 30, 30)
    assert L.scan(conn, NOW, only=[t])["weakened"] == 0, "it turned weak once; an hourly scan does not re-announce it"
    assert row(conn, t.key)[0][1] is False


def test_a_reversal_on_a_linked_sibling_row_counts(world):
    conn, t = world
    cases = _case_ids(30)
    # 30 cases at 510..800, two linked rows each; case j is decision j (full spread): exactly band 1.
    ids = fire(conn, t.key, [BASE[p // 2] for p in range(60)], decision_ids=[cases[p // 2] for p in range(60)],
               at=[p // 2 for p in range(60)])
    cond = t.form["hidden"]["condition"]
    before = LP.propose(cond, L.decisions(conn.cursor(), t.key, [1]), CFG)
    assert [(p["to"], p["stats"]["decisions"]) for p in before] == [(800, 30)], "without the reversal: band 1"
    _reverse(conn, ids[2 * 7 + 1])           # the SECOND row of case 7 (amount 580); the first is the one read
    decided = L.decisions(conn.cursor(), t.key, [1])
    assert len(decided) == 30 and [d.firing_id for d in decided if d.reversed] == [ids[2 * 7]]
    counts = L.scan(conn, NOW, only=[t])
    assert row(conn, t.key) == [] and counts["raised"] == 0 and counts["errors"] == 0


def _hidden_until_thirty_more(world, status):
    conn, t = world
    fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    _set_status(conn, t.key, status, 30)     # the person acted when the card showed 30 decisions
    # 26 more approvals, i = 33..58, 840..1090: 59 in all. Band 1 = 510..800; the other 29 are one
    # short of a band. Still to 800 on 30 decisions; 30 < 30 + 30, so the card stays hidden.
    fire(conn, t.key, [510 + 10 * i for i in range(33, 59)], first=33)
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["reopened"], counts["updated"], counts["raised"]) == (0, 0, 0)
    assert [(r[0], r[4], r[2]["to"]) for r in row(conn, t.key)] == [(status, 30, 800)]
    # one more, i = 59, 1100: band 2 = i 30..59 = 810..1100, 30 consecutive i, so it passes alone.
    # to = 1100 (two significant figures already); released (500, 1100] = 60 >= 30 + 30.
    fire(conn, t.key, [1100], first=59)
    counts = L.scan(conn, NOW, only=[t])
    assert counts["reopened"] == 1 and counts["errors"] == 0
    [(now_status, confident, proposal, stats, n_at_action)] = row(conn, t.key)
    assert (now_status, confident, proposal["to"], proposal["bands"], stats["decisions"], n_at_action) == \
        ("open", True, 1100, 2, 60, None)


def test_not_yet_hides_until_thirty_more_decisions(world):
    _hidden_until_thirty_more(world, "not_yet")


def test_dismissed_returns_only_on_materially_stronger_evidence(world):
    _hidden_until_thirty_more(world, "dismissed")


def _latest(target, value):
    """The same live policy with a newer version beside it whose limit is `value`."""
    condition = copy.deepcopy(target.form["hidden"]["condition"])
    condition["all"][1]["value"] = value
    return L.Target(key=target.key, live_version=target.live_version, form=target.form, versions=target.versions,
                    latest_condition=condition)


def _saved(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT status, saved_version, n_at_action FROM proc.bp_agent_policy_suggestion "
                "WHERE policy_key = %s ORDER BY suggestion_id", (key,))
    return cur.fetchall()


def test_a_saved_suggestion_stays_saved_while_a_version_in_force_holds_it_and_returns_when_none_does(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = 3 "
                          "WHERE policy_key = %s", (t.key,))
    # 40 more, i = 33..72, 840..1230: 73 decisions. Bands 510..800 and 810..1100 pass, the 13 above
    # are a tail -> to 1100 on 60, which would reopen anything but a saved card.
    fire(conn, t.key, [510 + 10 * i for i in range(33, 73)], first=33)
    # The latest version (a draft beside the live one) holds the change: 800, or 600 typed by hand,
    # are both looser than 500 in the suggested direction. Saved means saved.
    for held in (800, 600):
        counts = L.scan(conn, NOW, only=[_latest(t, held)])
        assert (counts["raised"], counts["updated"], counts["reopened"], counts["errors"]) == (0, 0, 0, 0), held
        [(status, _c, proposal, stats, _n)] = row(conn, t.key)
        assert (status, proposal["to"], stats["decisions"]) == ("saved", 800, 30)
        assert _saved(conn, t.key) == [("saved", 3, None)]
    # The draft was saved back at 500 (or replaced): no version in force holds the change, live or
    # latest. Nobody dismissed the card, so it returns, with the evidence as it now stands.
    counts = L.scan(conn, NOW, only=[_latest(t, 500)])
    assert (counts["reopened"], counts["raised"], counts["updated"], counts["errors"]) == (1, 0, 0, 0)
    [(status, confident, proposal, stats, _n)] = row(conn, t.key)
    assert (status, confident, proposal["to"], proposal["bands"], stats["decisions"]) == ("open", True, 1100, 2, 60)
    assert _saved(conn, t.key) == [("open", None, None)], "the version it was saved in is no longer the story"


def test_a_target_with_no_newer_version_is_judged_on_its_live_condition(world):
    conn, t = world                       # an injected target: latest_condition not given
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = 2 "
                          "WHERE policy_key = %s", (t.key,))
    assert L.scan(conn, NOW, only=[t])["reopened"] == 1      # live is 500 and nothing newer holds 800
    assert _saved(conn, t.key) == [("open", None, None)]


def test_a_saved_suggestion_with_no_passing_proposal_is_left_alone(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = 2 "
                          "WHERE policy_key = %s", (t.key,))
    # a rejection at 605: band 1 is 29 of 30 (Wilson 0.8333 < 0.85), so nothing is proposed this scan
    fire(conn, t.key, [605], result="rejected", first=33)
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["reopened"], counts["weakened"], counts["superseded"]) == (0, 0, 0)
    assert _saved(conn, t.key) == [("saved", 2, None)]


def test_a_changed_condition_supersedes_the_card(world):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    over_700 = _limit(t, 700)
    # limit 700: the evidence is 710..830, 13 decisions, not a band -> nothing proposed for 700,
    # and the card about 500 no longer describes the condition.
    counts = L.scan(conn, NOW, only=[over_700])
    assert (counts["superseded"], counts["raised"], counts["errors"]) == (1, 0, 0)
    assert {k: v["status"] for k, v in by_change(conn, t.key).items()} == {CHANGE_500: "superseded"}
    # 17 more, i = 33..49, 840..1000. Above 700 there are now 710..1000 = i 20..49 = 30 consecutive
    # i: one band, passes alone -> to 1000, released (700, 1000] = 30.
    fire(conn, t.key, [510 + 10 * i for i in range(33, 50)], first=33)
    counts = L.scan(conn, NOW, only=[over_700])
    assert (counts["raised"], counts["superseded"]) == (1, 0)
    cards = by_change(conn, t.key)
    new = cards["threshold:args.amount:gt:700"]
    assert (new["status"], new["proposal"]["from"], new["proposal"]["to"], new["stats"]["decisions"]) == \
        ("open", 700, 1000, 30)
    assert cards[CHANGE_500]["status"] == "superseded"
    # back to 500: 50 decisions 510..1000, band 1 = 510..800, the other 20 are not a band -> to 800.
    # The superseded card is raised again; the one about 700 is superseded in its turn.
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["raised"], counts["superseded"]) == (1, 1)
    cards = by_change(conn, t.key)
    assert (cards[CHANGE_500]["status"], cards[CHANGE_500]["proposal"]["to"]) == ("open", 800)
    assert cards["threshold:args.amount:gt:700"]["status"] == "superseded"


def test_evidence_from_an_older_version_with_the_same_condition_counts(world):
    conn, t = world
    fire(conn, t.key, BASE[:20], version=1)
    fire(conn, t.key, BASE[20:], version=2, first=20)
    live_only = L.Target(key=t.key, live_version=2, form=t.form, versions=[2])
    assert L.scan(conn, NOW, only=[live_only])["raised"] == 0, "13 decisions under version 2 alone"
    both = L.Target(key=t.key, live_version=2, form=t.form, versions=[1, 2])
    assert L.scan(conn, NOW, only=[both])["raised"] == 1
    card = by_change(conn, t.key)[CHANGE_500]
    # the same 33 decisions as the base data: band 1 = 510..800 -> to 800 on 30
    assert (card["proposal"]["to"], card["stats"]["decisions"], card["version"]) == (800, 30, 2)


def test_one_broken_policy_does_not_stop_the_others(world):
    conn, t = world
    _, b, c = _siblings(t.key)
    nonsense = copy.deepcopy(t.form)
    nonsense["hidden"]["condition"] = "nonsense"          # nothing to loosen: not an error
    unreadable = L.Target(key=b, live_version=1, form=None, versions=[1])      # an exception
    good = L.Target(key=c, live_version=1, form=t.form, versions=[1])
    fire(conn, t.key, BASE)
    fire(conn, b, BASE)
    fire(conn, c, BASE)
    counts = L.scan(conn, NOW, only=[L.Target(key=t.key, live_version=1, form=nonsense, versions=[1]),
                                     unreadable, good])
    assert (counts["policies"], counts["errors"], counts["raised"]) == (3, 1, 1)
    assert row(conn, t.key) == [] and row(conn, b) == []
    assert [(r[0], r[2]["to"]) for r in row(conn, c)] == [("open", 800)]
    assert conn.autocommit is True, "the failed policy's transaction was closed"


def _learning_log(caplog):
    return [r for r in caplog.records if r.name == L.logger.name]


@pytest.mark.parametrize("key,value", [("min_decisions", "thirty"), ("dismiss_more", None),
                                       ("wilson_lower", float("nan")), ("min_days", -1), ("min_approvers", True)])
def test_a_non_numeric_setting_raises_no_suggestion(world, monkeypatch, caplog, key, value):
    conn, t = world
    _, b, _c = _siblings(t.key)
    fire(conn, t.key, BASE)
    fire(conn, b, BASE)
    monkeypatch.setattr(S, "load_settings", lambda conn=None: S.merge({"learning": {key: value}}))
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[t, L.Target(key=b, live_version=1, form=t.form, versions=[1])])
    assert row(conn, t.key) == [] and row(conn, b) == []
    assert counts == {**dict.fromkeys(L.COUNTERS, 0), "errors": 1}, "one error for the scan, not one per policy"
    [record] = _learning_log(caplog)
    assert record.levelno == logging.WARNING and record.exc_info is None and key in record.getMessage()


def _setting(monkeypatch, **learning):
    monkeypatch.setattr(S, "load_settings", lambda conn=None: S.merge({"learning": learning}))


def test_a_dismissed_card_does_not_return_when_the_condition_goes_away_and_comes_back(world):
    conn, t = world
    fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    _set_status(conn, t.key, "dismissed", 30)             # dismissed when the card showed 30 decisions
    # limit 700: 13 decisions above it, nothing proposed; the dismissed card no longer fits -> superseded
    assert L.scan(conn, NOW, only=[_limit(t, 700)])["superseded"] == 1
    assert [(r[0], r[4]) for r in row(conn, t.key)] == [("superseded", 30)], "it remembers it was dismissed at 30"
    # limit back to 500: the same 33 decisions propose 800 on 30 again. 30 < 30 + max(30, 30): no new
    # evidence since the owner said no, so it must not come back.
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["raised"], counts["reopened"], counts["updated"]) == (0, 0, 0)
    assert [(r[0], r[4]) for r in row(conn, t.key)] == [("superseded", 30)]
    # 27 more, i = 33..59, 840..1100: 60 in all, bands 510..800 and 810..1100 -> to 1100 on 60 >= 30 + 30
    fire(conn, t.key, [510 + 10 * i for i in range(33, 60)], first=33)
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["reopened"], counts["raised"]) == (1, 0), "a hidden card coming back is a reopen"
    [(status, confident, proposal, stats, n_at_action)] = row(conn, t.key)
    assert (status, confident, proposal["to"], stats["decisions"], n_at_action) == ("open", True, 1100, 60, None)


def test_the_longer_of_the_two_waits_applies_once_superseded(world, monkeypatch):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    _set_status(conn, t.key, "superseded", 30)            # was not_yet or dismissed: no longer known
    # 27 more: 60 decisions, to 1100 on 60. not_yet_more 30 would let it back (60 >= 60);
    # dismiss_more 31 does not (60 < 61). The longer wait decides.
    fire(conn, t.key, [510 + 10 * i for i in range(33, 60)], first=33)
    _setting(monkeypatch, dismiss_more=31)
    assert L.scan(conn, NOW, only=[t])["reopened"] == 0 and row(conn, t.key)[0][0] == "superseded"
    _setting(monkeypatch, dismiss_more=30)
    assert L.scan(conn, NOW, only=[t])["reopened"] == 1 and row(conn, t.key)[0][0] == "open"


@pytest.mark.parametrize("status", ["not_yet", "dismissed"])
def test_a_hidden_card_with_no_decision_count_stays_hidden_and_says_so(world, caplog, status):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    _set_status(conn, t.key, status, None)
    cur = conn.cursor()
    cur.execute("SELECT suggestion_id FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (t.key,))
    suggestion_id = cur.fetchone()[0]
    # 40 more: 73 decisions, to 1100 on 60. Counting the missing number as 0 would reopen it (60 >= 30).
    fire(conn, t.key, [510 + 10 * i for i in range(33, 73)], first=33)
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[t])
    assert (counts["reopened"], counts["updated"], counts["errors"]) == (0, 0, 0)
    assert [(r[0], r[2]["to"]) for r in row(conn, t.key)] == [(status, 800)]
    [record] = _learning_log(caplog)
    assert record.levelno == logging.WARNING and str(suggestion_id) in record.getMessage()


def test_past_the_cap_the_newest_decisions_are_the_ones_read(world, monkeypatch, caplog):
    conn, t = world
    monkeypatch.setattr(L, "MAX_DECISIONS", 30)
    # Written FIRST (lowest firing id) but decided LAST: a rejection at 600, asked yesterday.
    [rejected] = fire(conn, t.key, [600], result="rejected", asked_at=NOW - timedelta(days=1))
    ids = fire(conn, t.key, BASE[:29])                    # i 0..28, 510..790: 30 decisions in all
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        assert len(L.decisions(conn.cursor(), t.key, [1])) == 30
    assert _learning_log(caplog) == [], "exactly at the cap is not past it"
    fire(conn, t.key, [800], first=29)                    # i 29: 31 decisions, one more than the cap
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        read = [d.firing_id for d in L.decisions(conn.cursor(), t.key, [1])]
    # The oldest decision is i 0 (slot 0, 45 days ago); it is the one left out, not the rejection.
    assert len(read) == 30 and rejected in read and ids[0] not in read
    assert read[0] == rejected, "newest first"
    [record] = _learning_log(caplog)
    assert t.key in record.getMessage()
    # Read: 520..800 approved (29) and the rejection at 600 = one band of 30 with 29 approved
    # (Wilson 0.8333 < 0.85) -> nothing. Had the rejection been cut off: 510..800, 30 of 30 -> 800.
    counts = L.scan(conn, NOW, only=[t])
    assert row(conn, t.key) == [] and counts["errors"] == 0


def test_a_weak_card_recovers_when_the_evidence_does(world, monkeypatch):
    conn, t = world
    fire(conn, t.key, BASE)
    assert L.scan(conn, NOW, only=[t])["raised"] == 1
    _setting(monkeypatch, min_approvers=4)                # the company now wants 4 people; there are 3
    assert L.scan(conn, NOW, only=[t])["weakened"] == 1 and row(conn, t.key)[0][1] is False
    _setting(monkeypatch, min_approvers=3)
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["updated"], counts["raised"], counts["reopened"], counts["weakened"]) == (1, 0, 0, 0)
    [(status, confident, proposal, stats, _n)] = row(conn, t.key)
    assert (status, confident, proposal["to"], stats["decisions"]) == ("open", True, 800, 30)


def test_a_value_people_always_approve_is_suggested_for_removal_from_a_list(world):
    conn, t = world
    form = copy.deepcopy(t.form)
    form["hidden"]["condition"]["all"].append({"field": "customer.country", "op": "in", "value": ["FR", "DE"]})
    form["hidden"]["inputs"].append({"name": "Customer country", "field": "customer.country", "type": "string"})
    t = L.Target(key=t.key, live_version=1, form=form, versions=[1])
    # The base 33: i 0..29 (510..800) are FR, i 30..32 are DE. FR = 30 consecutive i, all approved,
    # 40 days, 3 people: passes. DE = 3 decisions: no. The amounts also give the threshold card (800 on 30).
    fire(conn, t.key, BASE, extra=[{"customer.country": "FR" if i < 30 else "DE"} for i in range(33)])
    counts = L.scan(conn, NOW, only=[t])
    assert (counts["raised"], counts["errors"]) == (2, 0)
    cards = by_change(conn, t.key)
    assert set(cards) == {CHANGE_500, "list:customer.country:FR"}
    card = cards["list:customer.country:FR"]
    assert (card["status"], card["confident"], card["proposal"]["kind"], card["proposal"]["remove"]) == \
        ("open", True, "list", "FR")
    assert card["proposal"]["plain"] == "Stop asking for approval when Customer country is FR?"
    assert (card["stats"]["decisions"], card["stats"]["approved"]) == (30, 30)
    cur = conn.cursor()
    cur.execute("SELECT kind FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s AND change_key = %s",
                (t.key, "list:customer.country:FR"))
    assert cur.fetchone()[0] == "list"
    # a rejection for FR: the group is 30 of 31 (Wilson 0.8382 < 0.85) -> the list card turns weak
    fire(conn, t.key, [505], result="rejected", first=33, extra=[{"customer.country": "FR"}])
    L.scan(conn, NOW, only=[t])
    weak = by_change(conn, t.key)["list:customer.country:FR"]
    assert (weak["confident"], weak["stats"]["decisions"], weak["stats"]["approved"]) == (False, 31, 30)


def test_cards_of_policies_no_longer_scanned_are_closed_and_no_others():
    """The one statement the unscoped scan ends with, run by hand in a transaction that is ROLLED BACK.
    Its key list is every key in the table plus nothing of ours, so the only rows it can match are the
    throwaway ones inserted in the same transaction. Nothing in the shared table changes."""
    gone, kept = f"ZZT-{uuid.uuid4().int % 10 ** 9 + 10 ** 9}", f"ZZT-{uuid.uuid4().int % 10 ** 9 + 2 * 10 ** 9}"
    test_key = f"TST-{uuid.uuid4().hex[:6]}a"
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT policy_key FROM proc.bp_agent_policy_suggestion")
            everyone = [r[0] for r in cur.fetchall()]
            rows = [(gone, "a", "open"), (gone, "b", "not_yet"), (gone, "c", "dismissed"), (gone, "d", "saved"),
                    (kept, "a", "open"), (test_key, "a", "open")]
            for key, change, status in rows:
                cur.execute("INSERT INTO proc.bp_agent_policy_suggestion (policy_key, change_key, policy_version, "
                            "kind, proposal, stats, confident, status) VALUES (%s,%s,1,'threshold','{}','{}',true,%s)",
                            (key, change, status))
            cur.execute(L._GONE_SQL, (NOW, everyone + [kept]))
            assert cur.rowcount == 3, "only the throwaway policy's three live cards"
            cur.execute("SELECT policy_key, change_key, status, updated_at FROM proc.bp_agent_policy_suggestion "
                        "WHERE policy_key = ANY(%s)", ([gone, kept, test_key],))
            after = {(r[0], r[1]): (r[2], r[3]) for r in cur.fetchall()}
            assert {k: v[0] for k, v in after.items()} == {
                (gone, "a"): "superseded", (gone, "b"): "superseded", (gone, "c"): "superseded",
                (gone, "d"): "saved", (kept, "a"): "open", (test_key, "a"): "open"}
            assert after[(gone, "a")][1] == NOW
        finally:
            conn.rollback()
            conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM proc.bp_agent_policy_suggestion WHERE policy_key = ANY(%s)",
                    ([gone, kept, test_key],))
        assert cur.fetchone()[0] == 0, "rolled back"


# ------------------------------------------------------------------ one unreadable row, one unreadable version
def _plant(conn, key, change_key, proposal, status="open"):
    cur = conn.cursor()
    cur.execute("INSERT INTO proc.bp_agent_policy_suggestion (policy_key, change_key, policy_version, kind, "
                "proposal, stats, confident, status) VALUES (%s, %s, 1, 'threshold', %s::jsonb, '{}'::jsonb, TRUE, %s) "
                "RETURNING suggestion_id", (key, change_key, json.dumps(proposal), status))
    return cur.fetchone()[0]


def test_one_unreadable_suggestion_row_is_skipped_and_the_policys_other_rows_are_still_processed(world, caplog):
    conn, t = world
    # Planted BEFORE the real card, so they come first under the lock's order. Neither is proposed
    # by the scan, so both are read in the "no longer proposed" pass:
    no_field = _plant(conn, t.key, "tst:no-field", {"kind": "threshold", "op": "gt", "from": 500})     # targets() fails
    no_to = _plant(conn, t.key, "tst:no-to", {"kind": "threshold", "field": "args.amount", "op": "gt",
                                              "from": 500})                                           # group_for() fails
    _plant(conn, t.key, "threshold:args.amount:gt:400",
           {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 400, "to": 450})      # readable: 400 is gone
    fire(conn, t.key, BASE)
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[t])
    # the policy's step did not fail: the real card is raised and the stale one closed
    assert (counts["errors"], counts["unreadable"], counts["raised"], counts["superseded"]) == (0, 2, 1, 1)
    got = by_change(conn, t.key)
    assert (got[CHANGE_500]["status"], got[CHANGE_500]["proposal"]["to"]) == ("open", 800)
    assert got["threshold:args.amount:gt:400"]["status"] == "superseded"
    assert (got["tst:no-field"]["status"], got["tst:no-to"]["status"]) == ("open", "open"), "left exactly as they were"
    said = [r.getMessage() for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(said) == 2 and all("KeyError" in m for m in said)
    assert f"suggestion {no_field} " in said[0] and f"suggestion {no_to} " in said[1]


def test_a_latest_version_whose_condition_is_not_a_condition_leaves_a_saved_row_alone(world, caplog):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = 2 "
                          "WHERE policy_key = %s", (t.key,))
    unreadable = L.Target(key=t.key, live_version=1, form=t.form, versions=[1], latest_condition="nonsense")
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[unreadable])
    # whether the newer version holds the change cannot be told: that is not "it does not"
    assert (counts["reopened"], counts["errors"], counts["unreadable"]) == (0, 0, 0)
    assert _saved(conn, t.key) == [("saved", 2, None)]
    assert len([r for r in caplog.records if t.key in r.getMessage() and "in force" in r.getMessage()]) == 1


def test_a_latest_condition_that_cannot_be_compared_does_not_fail_the_policys_step(world, monkeypatch, caplog):
    conn, t = world
    fire(conn, t.key, BASE)
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = 2 "
                          "WHERE policy_key = %s", (t.key,))
    held = _latest(t, 800)
    real = LP.applied

    def applied(proposal, condition):
        if condition == held.latest_condition:
            raise ValueError("TST cannot compare")
        return real(proposal, condition)
    monkeypatch.setattr(LP, "applied", applied)
    with caplog.at_level(logging.WARNING, logger=L.logger.name):
        counts = L.scan(conn, NOW, only=[held])
    assert (counts["errors"], counts["reopened"], counts["policies"]) == (0, 0, 1)
    assert _saved(conn, t.key) == [("saved", 2, None)]
    told = [r.getMessage() for r in caplog.records if t.key in r.getMessage() and "in force" in r.getMessage()]
    assert len(told) == 1 and "ValueError" in told[0], "once per policy, with the kind of failure"
