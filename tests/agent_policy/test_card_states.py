"""Honest card states and messages (stage 5b, Task B5). No database.

A card whose change cannot be drafted right now says why before anyone clicks (`blocked`, and
`canDraft` false); a second review that cannot be checked because a stored form cannot be read
says so (it is not "try again shortly"); and the policy page says who may record a review.
"""
import copy
import json
import logging

import psycopg2
import pytest

from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from services.agent_policy import settings as S
from tests.agent_policy.test_learning_cards_live import _row
from tests.agent_policy.test_second_review import COMPILED, FINANCE, _Broken, _Scripted, at, form

DRAFT_IN_THE_WAY = "The latest draft of this policy already changes this. Activate or discard that draft first."
NOT_ACTIVE = "This policy is no longer Active."
UNREADABLE = {"field": "secondReview", "code": "second_review_unreadable",
              "message": "This policy's saved form could not be read, so a second review cannot be checked. "
                         "An administrator needs to look at it."}
UNAVAILABLE = {"field": "secondReview", "code": "second_review_unavailable",
               "message": "Can't check whether a second review is needed right now. Try again shortly."}
WHO = "An approver who is not this policy's owner and has not saved this draft."


# ------------------------------------------------------------------ the card
def _card(limit=500, *, status="live", live=2, latest=2, **over):
    """A card on "amount over 500 -> 800" whose policy's LATEST form says `limit`."""
    row = _row(policy_status=status, live_version=live, latest_version=latest, **over)
    if row["form"] is not None:
        row["form"]["hidden"]["condition"]["all"][1]["value"] = limit
    return L.card(row)


def test_a_card_that_can_be_drafted_says_nothing_more():
    for c in (_card(), _card(latest=3)):               # live is the latest; a draft that still says 500
        assert c["canDraft"] is True and "blocked" not in c


@pytest.mark.parametrize("limit", [400, 700, 800, 500.5, "500", None])
def test_a_newer_draft_that_no_longer_starts_where_the_change_starts_blocks_it(limit):
    c = _card(limit, latest=3)
    assert (c["canDraft"], c["blocked"]) == (False, DRAFT_IN_THE_WAY)


def test_the_live_version_is_not_a_draft_in_the_way():
    # live IS the latest and no longer says 500: the next scan supersedes the card; no draft to activate or discard
    c = _card(400)
    assert "blocked" not in c


def test_a_draft_whose_condition_cannot_be_read_blocks_as_draft_would_refuse():
    row = _row(policy_status="live", live_version=2, latest_version=3)
    row["form"]["hidden"] = "nothing"
    assert L.card(row)["blocked"] == DRAFT_IN_THE_WAY
    assert L.card(_row(policy_status="live", live_version=2, latest_version=3, form=None))["blocked"] == DRAFT_IN_THE_WAY


@pytest.mark.parametrize("status", ["retired", "draft"])
def test_a_policy_that_is_no_longer_active_says_so(status):
    for latest in (2, 3):
        c = _card(status=status, live=None if status == "retired" else 2, latest=latest)
        assert (c["canDraft"], c["blocked"]) == (False, NOT_ACTIVE)
    # and it is said rather than the draft sentence when both hold
    assert _card(800, status="retired", latest=3)["blocked"] == NOT_ACTIVE


def test_a_weak_card_says_it_too_and_a_card_that_is_not_open_says_nothing():
    weak = _card(400, latest=3, confident=False)
    assert (weak["canDraft"], weak["suggestionConfidence"], weak["blocked"]) == (False, "Not enough yet",
                                                                                 DRAFT_IN_THE_WAY)
    for status in ("not_yet", "dismissed", "saved", "superseded"):
        row = _row(policy_status="retired", live_version=None, latest_version=3, status=status)
        c = L.card(row)
        assert c["canDraft"] is False and "blocked" not in c, status


def test_a_card_read_without_its_policy_row_is_as_it_was():
    """Only a test's injected suggestion has no policy row; what is not known is not said."""
    assert "blocked" not in L.card(_row()) and L.card(_row())["canDraft"] is True
    assert "blocked" not in L.card(_row(policy_status=None, live_version=None, latest_version=None))


def test_the_queue_reads_what_blocked_needs_in_its_one_statement():
    sql = " ".join(L._CARD_SQL.split())
    assert "p.live_version" in sql.split(" FROM ")[0] and sql.count("JOIN") == 3, "no further join, no query per card"
    r = (12, "FIN-0012", "open", "threshold", {}, {}, True, None, None, "live", 3, None, "CFO", 2)
    got = L._card_row(r)
    assert (got["policy_status"], got["latest_version"], got["live_version"]) == ("live", 3, 2)


# ------------------------------------------------------------------ the second review: unreadable is not "try again"
def _problems(*answers, base=2, new=None):
    cur = _Scripted(list(answers))
    cur.connection = None
    return SR.activation_problems(cur, "FIN-0001", 2, base, new if new is not None else form(at(800)))


@pytest.mark.parametrize("name,stored", [
    ("the live form is not JSON", [("{not json", COMPILED)]),
    ("the live form is a list", [([], COMPILED)]),
    ("the live form is null", [(None, COMPILED)]),
    ("the live version row is not there", []),
])
def test_a_live_form_that_cannot_be_read_is_a_permanent_fault_in_its_own_words(caplog, name, stored):
    with caplog.at_level(logging.ERROR, logger=SR.__name__):
        got = _problems([(7,)], stored)
    assert got == [UNREADABLE], name
    assert "routeTo" not in got[0], "the form would add a second 'administrator' sentence"
    assert any("FIN-0001" in r.getMessage() for r in caplog.records), "said, with the policy"


def test_a_draft_form_that_cannot_be_read_is_the_same_fault(monkeypatch):
    monkeypatch.setattr(SR, "load_settings", lambda conn: S.merge(None))
    reviewed_but_unreadable = [("{not json", "tst-reviewer", None, None)]
    assert _problems([(7,)], [(form(), COMPILED)], *FINANCE, *FINANCE, reviewed_but_unreadable) == [UNREADABLE]
    # the form being activated is not a form: leaving protection cannot be ruled out, so a review is required
    assert _problems([(7,)], [(form(), COMPILED)], *FINANCE, [], new="nothing") == [SR.REQUIRED]


@pytest.mark.parametrize("exc", [RuntimeError("relation cannot be read"), psycopg2.OperationalError("server closed"),
                                 psycopg2.errors.UndefinedTable("no such table"), OSError("network"),
                                 psycopg2.InterfaceError("connection already closed")])
def test_a_failed_database_read_still_says_try_again_shortly(exc):
    class _Failing(_Broken):
        def execute(self, sql, params=None):
            raise exc
    assert SR.activation_problems(_Failing(), "FIN-0001", 2, 3, form(at(800))) == [UNAVAILABLE]


def test_the_two_problems_are_copies_never_the_shared_constant():
    first = _problems([(7,)], [])
    first[0]["message"] = "changed by a caller"
    assert _problems([(7,)], []) == [UNREADABLE] and SR.UNREADABLE == UNREADABLE
    assert SR.REFUSALS["second_review_unreadable"] == UNREADABLE["message"]


# ------------------------------------------------------------------ recording a review of an unreadable form
class _Conn:
    autocommit = True

    def __init__(self, answers):
        self.cur = _Scripted(answers)
        self.cur.connection = self
        self.cur.__enter__ = lambda: self.cur
        self.done = []

    def cursor(self):
        cur = self.cur

        class _Ctx:
            def __enter__(self): return cur
            def __exit__(self, *a): return False
        return _Ctx()

    def commit(self): self.done.append("commit")
    def rollback(self): self.done.append("rollback")


def _record(answers):
    from datetime import datetime, timezone

    from api.auth import Principal
    conn = _Conn(answers)
    who = Principal(subject="tst-reviewer", email="r@example.test", claims={})
    with pytest.raises(L.Refused) as refused:
        SR.record(conn, "FIN-0001", 3, principal=who, note=None, now=datetime(2026, 10, 10, tzinfo=timezone.utc))
    assert conn.done == ["rollback"], "nothing is written"
    assert not any("INSERT" in q for q in conn.cur.asked)
    return refused.value.code, refused.value.message


def test_no_review_is_recorded_of_a_form_that_cannot_be_read_and_it_is_said_as_permanent():
    head, suggestion = [(2, 3)], [(7,)]
    # the draft's own stored form; then the live version's
    assert _record([head, [("draft", "tst-saver", "{not json")]]) == ("second_review_unreadable", UNREADABLE["message"])
    assert _record([head, [("draft", "tst-saver", form(at(800)))], suggestion, [("{not json", COMPILED)]]) == (
        "second_review_unreadable", UNREADABLE["message"])
    assert _record([head, [("draft", "tst-saver", form(at(800)))], suggestion, []]) == (
        "second_review_unreadable", UNREADABLE["message"])


def test_recording_still_says_try_again_when_the_database_fails(monkeypatch):
    def boom(conn):
        raise RuntimeError("the map cannot be read")
    monkeypatch.setattr(SR.deciders, "load_map", boom)
    monkeypatch.setattr(SR, "load_settings", lambda conn: S.merge(None))
    answers = [[(2, 3)], [("draft", "tst-saver", form(at(800)))], [(7,)], [(form(), COMPILED)], *FINANCE, *FINANCE,
               [("tst-saver",)]]
    assert _record(answers) == ("second_review_unavailable", UNAVAILABLE["message"])


# ------------------------------------------------------------------ the policy page: who may review
VERSIONS = [{"version": 2, "savedAs": "live", "form": form()}, {"version": 3, "savedAs": "draft", "form": form(at(800))}]
HEAD = {"liveVersion": 2, "latestVersion": 3}


def _status(monkeypatch, *review):
    monkeypatch.setattr(SR, "load_settings", lambda conn: S.merge(None))
    cur = _Scripted([[(7,)], [(form(), COMPILED)], *FINANCE, *FINANCE, list(review)])
    cur.connection = None
    return SR.status(cur, "FIN-0001", HEAD, copy.deepcopy(VERSIONS))


def test_a_required_review_not_yet_recorded_says_who_may_record_it(monkeypatch):
    assert _status(monkeypatch, (form(at(800)), None, None, None)) == {
        "required": True, "version": 3, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None,
        "whoMayReview": WHO}
    assert SR.WHO_MAY_REVIEW == WHO


def test_a_recorded_review_and_a_review_that_is_not_needed_do_not(monkeypatch):
    from datetime import datetime, timezone
    when = datetime(2026, 10, 10, 9, tzinfo=timezone.utc)
    got = _status(monkeypatch, (form(at(800)), "tst-reviewer", when, "r@example.test"))
    assert got == {"required": True, "version": 3, "reviewedBy": "tst-reviewer", "reviewedByName": "r@example.test",
                   "reviewedAt": "2026-10-10T09:00:00+00:00"}
    none = SR.status(_Broken(), "FIN-0001", {"liveVersion": 2, "latestVersion": 2}, VERSIONS[:1])
    assert none == {"required": False, "version": None, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}


def test_a_review_that_is_required_because_nothing_can_be_told_says_it_too(monkeypatch):
    got = SR.status(_Broken(), "FIN-0001", HEAD, VERSIONS)
    assert got == {"required": True, "version": 3, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None,
                   "whoMayReview": WHO}
    assert json.dumps(got)
