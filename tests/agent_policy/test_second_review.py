"""The second reviewer, without a database: which changes need one, and how the endpoint answers.

Numbers as everywhere in stage 5: the live policy is "args.amount gt 500" and learning suggested
500 -> 800. 600 is a smaller loosening in the same direction, 2000 goes beyond the suggestion,
400 is a tightening.
"""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR

LIVE = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                {"field": "args.amount", "op": "gt", "value": 500}]}


def at(limit):
    return {"all": [LIVE["all"][0], {"field": "args.amount", "op": "gt", "value": limit}]}


def sug(status="open", sid=7, frm=500, to=800):
    return {"id": sid, "status": status,
            "proposal": {"kind": "threshold", "field": "args.amount", "op": "gt", "from": frm, "to": to}}


# ------------------------------------------------------------------ loosening (pure)
def test_raising_the_limit_a_suggestion_pointed_at_needs_a_review_whatever_its_status():
    for status in ("open", "not_yet", "dismissed", "saved"):
        assert SR.loosening([sug(status)], LIVE, at(800)) == 7, status
    assert SR.loosening([sug()], LIVE, at(600)) == 7, "a smaller loosening in the same direction still counts"


def test_loosening_beyond_what_was_suggested_needs_a_review_too():
    assert SR.loosening([sug()], LIVE, at(2000)) == 7, "suggested 800, typed 2000"
    assert SR.loosening([sug()], LIVE, at(801)) == 7


def test_no_review_for_an_unchanged_or_tighter_condition_or_without_a_suggestion():
    assert SR.loosening([sug()], LIVE, LIVE) is None
    assert SR.loosening([sug()], LIVE, at(400)) is None
    assert SR.loosening([], LIVE, at(800)) is None
    assert SR.loosening([sug("superseded")], LIVE, at(800)) is None


def test_a_suggestion_about_an_earlier_limit_demands_nothing():
    # the row was raised when the live limit was 300; the live limit is now 500
    old = sug("dismissed", sid=3, frm=300, to=450)
    assert SR.loosening([old], LIVE, at(800)) is None, "it does not start from the live condition"
    assert SR.loosening([old, sug(sid=9)], LIVE, at(800)) == 9, "the one about the live condition does"


def test_the_first_suggestion_that_points_at_the_change_is_the_one_named():
    assert SR.loosening([sug("superseded", sid=1), sug("dismissed", sid=2), sug(sid=3)], LIVE, at(800)) == 2


def test_removing_a_listed_value_a_suggestion_pointed_at_needs_a_review():
    live = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                    {"field": "customer.country", "op": "in", "value": ["FR", "DE", "IT"]}]}

    def listed(*values):
        return {"all": [live["all"][0], {"field": "customer.country", "op": "in", "value": list(values)}]}

    fr = {"id": 5, "status": "open",
          "proposal": {"kind": "list", "field": "customer.country", "op": "in", "remove": "FR"}}
    assert SR.loosening([fr], live, listed("DE", "IT")) == 5
    assert SR.loosening([fr], live, listed("DE")) == 5, "FR gone, whatever else went with it"
    assert SR.loosening([fr], live, listed("FR", "DE")) is None, "another value removed: not the suggested change"
    assert SR.loosening([fr], live, live) is None
    assert SR.loosening([fr], listed("DE", "IT"), listed("DE")) is None, "FR is not in the live list any more"


def test_a_condition_that_cannot_be_read_as_one_loosens_nothing():
    assert SR.loosening([sug()], LIVE, None) is None
    assert SR.loosening([sug()], None, at(800)) is None
    assert SR.loosening([sug()], LIVE, {"any": [at(800)]}) is None


def test_a_stored_suggestion_that_cannot_be_read_is_an_error_not_a_pass():
    with pytest.raises(Exception):
        SR.loosening([{"id": 1, "status": "open", "proposal": {"kind": "threshold"}}], LIVE, at(800))
    with pytest.raises(Exception):
        SR.loosening([{"id": 1, "status": "open", "proposal": None}], LIVE, at(800))


# ------------------------------------------------------------------ fail closed (scripted cursor)
class _Broken:
    """A cursor whose every statement fails, as when a table cannot be read."""
    def __init__(self):
        self.asked = []

    def execute(self, sql, params=None):
        self.asked.append(" ".join(sql.split()))
        raise RuntimeError("relation cannot be read")


def test_activation_is_refused_when_the_check_itself_cannot_run(caplog):
    cur = _Broken()
    with caplog.at_level(logging.ERROR, logger=SR.__name__):
        problems = SR.activation_problems(cur, "FIN-0001", 2, 3, {"hidden": {"condition": at(800)}})
    assert problems == [{"field": "secondReview", "code": "second_review_unavailable",
                         "message": "Can't check whether a second review is needed right now. Try again shortly."}]
    assert len(cur.asked) == 1, "it stopped at the first statement that failed"
    assert any("FIN-0001" in r.getMessage() and r.exc_info for r in caplog.records), "said, with the policy"


def test_a_policy_never_made_live_is_not_looked_up_at_all():
    cur = _Broken()
    assert SR.activation_problems(cur, "FIN-0001", None, 1, {"hidden": {"condition": at(800)}}) == []
    assert cur.asked == [], "nothing is live, so nothing can be loosened: no statement"


class _Scripted:
    def __init__(self, answers):
        self.answers, self.asked, self._rows = list(answers), [], []

    def execute(self, sql, params=None):
        self.asked.append(" ".join(sql.split()))
        self._rows = self.answers.pop(0)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_a_policy_with_no_suggestion_costs_one_select_and_needs_nothing():
    cur = _Scripted([[]])
    assert SR.activation_problems(cur, "OPS-0001", 2, 2, {"hidden": {"condition": at(800)}}) == []
    assert len(cur.asked) == 1 and "bp_agent_policy_suggestion" in cur.asked[0]


def test_the_policy_page_says_required_when_it_cannot_tell(caplog):
    versions = [{"version": 2, "savedAs": "live", "form": {"hidden": {"condition": LIVE}}},
                {"version": 3, "savedAs": "draft", "form": {"hidden": {"condition": at(800)}}}]
    head = {"liveVersion": 2, "latestVersion": 3}
    with caplog.at_level(logging.ERROR, logger=SR.__name__):
        got = SR.status(_Broken(), "FIN-0001", head, versions)
    assert got == {"required": True, "version": 3, "reviewedBy": None, "reviewedAt": None}
    assert any("FIN-0001" in r.getMessage() for r in caplog.records)


def test_the_policy_page_asks_nothing_when_there_is_no_draft_on_a_live_version():
    none = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}
    live = {"version": 2, "savedAs": "live", "form": {"hidden": {"condition": LIVE}}}
    draft = {"version": 1, "savedAs": "draft", "form": {"hidden": {"condition": LIVE}}}
    retired = {"version": 3, "savedAs": "retired", "form": {"hidden": {"condition": LIVE}}}
    for head, versions in (({"liveVersion": 2, "latestVersion": 2}, [draft, live]),          # live is the latest
                           ({"liveVersion": None, "latestVersion": 1}, [draft]),              # never live
                           ({"liveVersion": None, "latestVersion": 3}, [draft, live, retired])):
        cur = _Broken()
        assert SR.status(cur, "FIN-0001", head, versions) == none
        assert cur.asked == [], "no statement: the page cannot fail on this"


# ------------------------------------------------------------------ the endpoint through the whole app
HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@example.test",
       "X-User-Groups": json.dumps(["PROCWISE_APPROVER"])}
URL = "/agent-policies/FIN-0012/second-review"
DONE = {"policyKey": "FIN-0012", "version": 3, "reviewedBy": "u1", "reviewedAt": "2026-10-10T09:00:00+00:00"}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


@pytest.fixture
def state():
    return {"role": "Approver", "audits": [], "calls": []}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: state["role"])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: state["audits"].append(kw))

    def record(conn, key, version, **kw):
        state["calls"].append((key, version, kw))
        return dict(DONE)
    monkeypatch.setattr(SR, "record", record)
    return TestClient(app)


def _refuse(code, message):
    def _raise(*a, **k):
        raise L.Refused(code, message)
    return _raise


def test_recording_a_review_answers_201_and_passes_who_and_what(client, state):
    r = client.post(URL, json={"version": 3, "note": "  checked the evidence  "}, headers=HDR)
    assert r.status_code == 201, r.text
    # not exempt from the output filter, and nothing in it is withheld: ids, a subject and an ISO time
    assert r.json() == DONE
    [(key, version, kw)] = state["calls"]
    assert (key, version, kw["note"]) == ("FIN-0012", 3, "  checked the evidence  ")
    assert (kw["principal"].subject, kw["principal"].email) == ("u1", "u1@example.test")
    assert [(a["action_type"], a["phase"], a["status"]) for a in state["audits"]] == [
        ("agent_policy.second_review", "authorize", "allowed"), ("agent_policy.second_review", "act", "done")]
    assert state["audits"][0]["details"]["policy"] == "FIN-0012" and state["audits"][0]["details"]["version"] == 3
    assert state["audits"][0]["details"]["minimum"] == "Approver"


@pytest.mark.parametrize("role", ["Viewer", "Buyer", "Nobody"])
def test_only_an_approver_reaches_the_review(client, state, role):
    state["role"] = role
    r = client.post(URL, json={"version": 3}, headers=HDR)
    assert r.status_code == 403
    assert state["calls"] == [], "refused before anything is looked at"
    assert [(a["phase"], a["status"]) for a in state["audits"]] == [("authorize", "denied")]


def test_the_gateway_key_is_required(client, state):
    r = client.post(URL, json={"version": 3}, headers={**HDR, "X-Gateway-Key": "wrong"})
    assert r.status_code == 401 and state["calls"] == [] and state["audits"] == []


@pytest.mark.parametrize("code,message", [
    ("own_change", "You saved this change, so someone else must review it."),
    ("owner", "The policy's owner cannot be its second reviewer."),
    ("owner_unmapped", "Nobody is linked to this policy's owner yet, so a second review cannot be recorded. "
                       "An administrator must link them first."),
    ("not_latest", "This is no longer the latest draft of the policy. Reload and try again."),
    ("not_needed", "This change does not need a second review."),
    ("already_reviewed", "This draft has already been reviewed."),
    ("second_review_unavailable", "Can't check whether a second review is needed right now. Try again shortly."),
])
def test_a_refusal_is_a_409_whose_code_and_message_arrive_readable(client, state, monkeypatch, code, message):
    # the real messages, raised by the real module's constants where it has them
    monkeypatch.setattr(SR, "record", _refuse(code, message))
    r = client.post(URL, json={"version": 3}, headers=HDR)
    assert r.status_code == 409, r.text
    assert r.json() == {"code": code, "message": message}, "the output filter left it readable"
    assert [(a["phase"], a["status"], a["details"].get("code")) for a in state["audits"]] == [
        ("authorize", "allowed", None), ("act", "refused", code)]


def test_the_messages_the_module_raises_are_the_ones_checked_above():
    assert SR.MESSAGE == ("A second reviewer must review this change before it goes Active. "
                          "It loosens the policy as learning suggested.")
    assert SR.REFUSALS == {
        "own_change": "You saved this change, so someone else must review it.",
        "owner": "The policy's owner cannot be its second reviewer.",
        "owner_unmapped": "Nobody is linked to this policy's owner yet, so a second review cannot be recorded. "
                          "An administrator must link them first.",
        "not_latest": "This is no longer the latest draft of the policy. Reload and try again.",
        "not_needed": "This change does not need a second review.",
        "already_reviewed": "This draft has already been reviewed.",
        "second_review_unavailable": "Can't check whether a second review is needed right now. Try again shortly.",
        "not_found": "No such policy."}


def test_an_unknown_policy_is_a_404(client, state, monkeypatch):
    monkeypatch.setattr(SR, "record", _refuse("not_found", "No such policy."))
    assert client.post(URL, json={"version": 3}, headers=HDR).status_code == 404
    # a key that is not a policy key never reaches the database
    state["calls"].clear()
    monkeypatch.setattr(SR, "record", lambda *a, **k: state["calls"].append(a) or dict(DONE))
    assert client.post("/agent-policies/suggestions/second-review", json={"version": 3}, headers=HDR).status_code == 404
    assert state["calls"] == []


@pytest.mark.parametrize("body", [{}, {"version": 0}, {"version": "three"}, {"version": 2 ** 31},
                                  {"version": 3, "note": "x" * 2001}])
def test_a_malformed_body_is_refused_before_anything_is_looked_at(client, state, body):
    assert client.post(URL, json=body, headers=HDR).status_code == 422
    assert state["calls"] == []


def test_a_failure_says_nothing_internal(client, state, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("relation proc.bp_agent_policy_second_review does not exist")
    monkeypatch.setattr(SR, "record", boom)
    r = client.post(URL, json={"version": 3}, headers=HDR)
    assert r.status_code == 500 and "bp_agent_policy" not in r.text and "RuntimeError" not in r.text
    assert [(a["phase"], a["status"]) for a in state["audits"]] == [("authorize", "allowed"), ("act", "error")]
