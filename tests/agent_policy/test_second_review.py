"""The second reviewer, without a database: the rule (which changes need one), and how the
endpoint answers.

The rule makes no judgement of direction: once learning has raised a suggestion on a protected
policy, ANY change to what it enforces needs a second person. "Changes" is decided on canonical
JSON text of the form's enforced keys.
"""
import copy
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from services.agent_policy import settings as S
from tests.agent_policy.fixtures import FORM_EXAMPLE

SETTINGS = S.merge(None)
TOOL = {"field": "tool.name", "op": "in", "value": ["refund.issue"]}
LIVE = {"all": [TOOL, {"field": "args.amount", "op": "gt", "value": 500}]}


def amount(op, value):
    return {"field": "args.amount", "op": op, "value": value}


def at(limit):
    return {"all": [TOOL, amount("gt", limit)]}


def form(condition=LIVE, *, outcome="approve", tools=("refund.issue",), on_missing=None, area="Finance",
         checkpoint="tool.call.before"):
    return {"outcome": outcome, "businessArea": area, "owner": "CFO",
            "hidden": {"checkpoint": checkpoint, "actions": {"tools": list(tools)}, "condition": condition,
                       "onMissingData": on_missing}}


def full(**changes):
    """The whole example form (every key a real form has), with top-level keys replaced."""
    return {**copy.deepcopy(FORM_EXAMPLE), **changes}


def hidden(**changes):
    return full(hidden={**copy.deepcopy(FORM_EXAMPLE["hidden"]), **changes})


def changed(new, live=None, settings=SETTINGS):
    return SR._changed(live if live is not None else full(), new, settings)


# ------------------------------------------------------------------ canonical text
def test_canonical_text_tells_apart_what_python_calls_equal():
    assert 1 == True and 800 == 800.0                                        # noqa: E712 - the point
    assert SR._canon(1) != SR._canon(True)
    assert SR._canon(800) != SR._canon(800.0)
    assert SR._canon({"a": None}) != SR._canon({})
    assert SR._canon({"a": 1, "b": [1, {"c": 2, "d": 3}]}) == SR._canon({"b": [1, {"d": 3, "c": 2}], "a": 1})
    assert SR._canon({"a": [1, 2]}) != SR._canon({"a": [2, 1]}), "the order of a list is part of it"
    assert SR._canon({"b": 1, "a": "é"}) == '{"a":"\\u00e9","b":1}'


# ------------------------------------------------------------------ the rule (pure)
def test_the_enforced_keys_are_exactly_these():
    assert SR.ENFORCED == ("outcome", "deciders", "notify", "responseTime", "limit", "hidden", "source",
                           "effectiveFrom")
    assert set(SR.ENFORCED) <= set(FORM_EXAMPLE), "each is a key a real form has"


SAME = {   # what a person may edit without a second person: none of it is enforced
    "resubmitted unchanged": full(),
    "the form through JSON, keys in another order": json.loads(json.dumps(full(), sort_keys=True)),
    "name": full(name="Another name"),
    "situation": full(situation="Reworded."),
    "the message for the agent": full(messageForAgent="Reworded."),
    "the message for the person": full(messageForPerson="Reworded."),
    "change note": full(changeNote="a note"),
    "owner": full(owner="Someone else"),
    "category": full(category="Another category"),
    "sub-area": full(subArea="Another sub-area"),
    "review-by date": full(reviewBy="2027-01-01"),
    # NOT here since fix round 4: the effective date (see DIFFERENT, "the date it starts")
    "examples": full(examples=[]),
    "the confirmation": full(checked={"by": "x", "at": "t"}),
    "the company's missing-data rule said out loud": hidden(onMissingData="fail_closed"),
}


@pytest.mark.parametrize("name", sorted(SAME))
def test_an_edit_to_what_is_not_enforced_is_no_change(name):
    assert changed(SAME[name]) is False


def _cond(*rest, top=None):
    tool = copy.deepcopy(FORM_EXAMPLE["hidden"]["condition"]["all"][0])
    condition = {"all": [tool, *rest]}
    return hidden(condition=top(condition) if top else condition)


EXISTS = {"field": "args.amount", "op": "exists"}
B_EXISTS, C_EXISTS = {"field": "args.b", "op": "exists"}, {"field": "args.c", "op": "exists"}
_inputs = copy.deepcopy(FORM_EXAMPLE["hidden"]["inputs"])
_inputs[0]["sensitive"] = True
DIFFERENT = {   # every route of the three review rounds, and each enforced key on its own
    "a loosening": _cond(amount("gt", 800)),
    "a tightening": _cond(amount("gt", 499)),
    "gte 800": _cond(amount("gte", 800)),
    "gt 500 AND gte 800": _cond(amount("gt", 500), amount("gte", 800)),
    "gt 800 listed twice": _cond(amount("gt", 800), amount("gt", 800)),
    "any[gt 800] inside the all": _cond({"any": [amount("gt", 800)]}),
    "not lte 800": _cond({"not": amount("lte", 800)}),
    "a top-level any wrapper": _cond(amount("gt", 500), top=lambda c: {"any": [c]}),
    "the same limit, written another way": _cond({"not": amount("lte", 500)}),
    "the same leaves in another order": hidden(condition={"all": FORM_EXAMPLE["hidden"]["condition"]["all"][::-1]}),
    "an exists guard": _cond(EXISTS, amount("gt", 500)),
    "not(not exists)": _cond({"not": {"not": EXISTS}}, amount("gt", 500)),
    "a value-dependent guard": _cond(amount("gt", 500), {"any": [EXISTS, {"field": "tool.name", "op": "eq",
                                                                          "value": "refund.issue"}]}),
    "a guard on some of the fields": _cond(amount("gt", 500), {"any": [EXISTS, B_EXISTS, {"not": C_EXISTS}]}),
    "500 written 500.0": _cond(amount("gt", 500.0)),
    "outcome": full(outcome="notify", notify=["Someone"]),
    "outcome alone": full(outcome="block"),
    "deciders": full(deciders=["CFO"]),
    "deciders in another order": full(deciders=["CFO", "Finance Manager"]),
    "notify": full(notify=["Someone"]),
    "response time": full(responseTime="PT1H"),
    "limit": full(limit={"on": True, "text": "UK only"}),
    # Fix round 4. Enforcement reads `source.document`: two policies from the same document are
    # "tiered" (each keeps its approval levels), from different ones "conflicting" (last level
    # only), so renaming it can skip a first-level decider. The whole key is compared.
    "the source document renamed": full(source={**FORM_EXAMPLE["source"], "document": "Another policy document"}),
    "the source reference": full(source={**FORM_EXAMPLE["source"], "reference": "9.9"}),
    # Was pinned as "no change" until fix round 4. It is published to the orchestrator as
    # `effective.from` and may be honoured outside this backend, so moving it is a change.
    "the date it starts": full(effectiveFrom="2026-03-01"),
    "checkpoint": hidden(checkpoint="agent.reply.before"),
    "a tool removed": hidden(actions={**FORM_EXAMPLE["hidden"]["actions"], "tools": ["credit.issue"]}),
    "a tool added": hidden(actions={**FORM_EXAMPLE["hidden"]["actions"], "tools": ["refund.issue", "credit.issue", "x"]}),
    "no action listed": hidden(actions={**FORM_EXAMPLE["hidden"]["actions"], "tools": []}),
    "an input's sensitive flag": hidden(inputs=_inputs),
    "units": hidden(units={**FORM_EXAMPLE["hidden"]["units"], "currency": "EUR"}),
    "time window": hidden(timeWindow={"days": ["sat"], "from": "18:00", "to": "08:00", "timeZone": "Europe/London"}),
    "fail open": hidden(onMissingData="fail_open"),
    "while paused": hidden(whilePaused="retry"),
    "a key under hidden nobody has thought of": hidden(somethingNew=1),
    "a key gone": {k: v for k, v in full().items() if k != "limit"},
    "no hidden at all": {k: v for k, v in full().items() if k != "hidden"},
    "hidden that is not an object": full(hidden="nothing"),
}


@pytest.mark.parametrize("name", sorted(DIFFERENT))
def test_any_difference_in_what_is_enforced_is_a_change(name):
    assert changed(DIFFERENT[name]) is True


def test_one_and_true_are_different_and_so_are_800_and_800_point_0():
    flag = {"field": "args.flag", "op": "eq", "value": 1}
    assert changed(_cond(amount("gt", 500), {**flag, "value": True}), live=_cond(amount("gt", 500), flag)) is True
    assert changed(_cond(amount("gt", 800.0)), live=_cond(amount("gt", 800))) is True
    assert changed(_cond(amount("gt", 800)), live=_cond(amount("gt", 800))) is False


def test_the_missing_data_rule_is_compared_as_it_is_enforced():
    """The live form names no rule and was compiled fail-closed; an Admin has since made the
    company default fail-open. Saving the same form again would compile it fail-open."""
    company = S.merge({"on_missing_data": {"approve": "fail_open"}})
    compiled = {"trigger": {"onMissingData": "fail_closed"}}
    live = SR._as_enforced(full(), compiled)
    assert live["hidden"]["onMissingData"] == "fail_closed" and full()["hidden"]["onMissingData"] is None
    assert changed(full(name="Reworded"), live=live, settings=company) is True, "it would newly fail open"
    assert changed(hidden(onMissingData="fail_closed"), live=live, settings=company) is False, "kept closed, out loud"
    assert changed(full(), live=SR._as_enforced(full(), {"trigger": {"onMissingData": "fail_open"}}),
                   settings=company) is False, "it already was open"
    # the document as stored text; and no compiled rule to read: the form is handed back as it is
    assert SR._as_enforced(full(), json.dumps(compiled)) == live
    for unreadable in (None, {}, {"trigger": {}}, {"trigger": {"onMissingData": None}}, "not a document"):
        assert SR._as_enforced(full(), unreadable) == full(), repr(unreadable)
        assert changed(full(name="Reworded"), live=SR._as_enforced(full(), unreadable), settings=company) is False


def test_a_review_covers_the_whole_form_but_for_the_confirmation_and_the_note():
    stored = {**form(at(800)), "checked": {"by": "a", "at": "t"}, "changeNote": "one", "deciders": ["CFO"]}
    assert SR._same_form(stored, {**stored, "checked": None, "changeNote": "two"})
    assert SR._same_form(stored, json.loads(json.dumps(stored, sort_keys=True)))
    for key, value in (("outcome", "notify"), ("deciders", ["Someone"]), ("owner", "Other"),
                       ("businessArea", "Operations"), ("messageForAgent", "reworded")):
        assert not SR._same_form(stored, {**stored, key: value}), key
    for key, value in (("onMissingData", "fail_open"), ("actions", {"tools": ["credit.issue"]}), ("condition", at(900)),
                       ("condition", at(800.0))):
        assert not SR._same_form(stored, {**stored, "hidden": {**stored["hidden"], key: value}}), key
    assert not SR._same_form(stored, None) and not SR._same_form(None, stored)


# ------------------------------------------------------------------ fail closed, and the cost (scripted cursor)
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
        problems = SR.activation_problems(cur, "FIN-0001", 2, 3, form(at(800)))
    assert problems == [{"field": "secondReview", "code": "second_review_unavailable",
                         "message": "Can't check whether a second review is needed right now. Try again shortly."}]
    assert len(cur.asked) == 1, "it stopped at the first statement that failed"
    assert any("FIN-0001" in r.getMessage() and r.exc_info for r in caplog.records), "said, with the policy"


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


@pytest.fixture
def no_history(monkeypatch):
    """The settings must not be read at all."""
    def boom(*a, **k):
        raise AssertionError("not for this policy")
    monkeypatch.setattr(SR, "load_settings", boom)


def test_a_policy_with_no_suggestion_costs_one_select_and_needs_nothing(no_history):
    for live in (2, None):                                   # live now, or never live / retired
        cur = _Scripted([[]])
        assert SR.activation_problems(cur, "OPS-0001", live, 2, form(at(800))) == []
        assert len(cur.asked) == 1 and "bp_agent_policy_suggestion" in cur.asked[0] and "LIMIT 1" in cur.asked[0]


def test_a_policy_outside_a_protected_area_costs_three_more_and_needs_nothing(no_history):
    ops = form(area="Operations")
    # the suggestion row, the live form, the area as a save resolves it, its setting
    cur = _Scripted([[(7,)], [(ops, None)], [("Operations", False)], [(False,)]])
    assert SR.activation_problems(cur, "OPS-0001", 2, 2, form(at(800), area="Operations")) == []
    assert len(cur.asked) == 4


FINANCE = [("Finance", False)], [(True,)]          # the area as a save resolves it, then its setting
COMPILED = {"trigger": {"onMissingData": "fail_closed"}}


def test_a_protected_policy_is_judged_without_ever_reading_the_decisions(monkeypatch):
    monkeypatch.setattr(SR, "load_settings", lambda conn: SETTINGS)

    def problems(new, *more):
        cur = _Scripted([[(7,)], [(form(), COMPILED)], *FINANCE, *more])
        cur.connection = None
        got = SR.activation_problems(cur, "FIN-0001", 2, 2, new)
        assert not any("bp_policy_firing" in q for q in cur.asked)
        return got, len(cur.asked)

    assert problems({**form(), "owner": "Reworded"}, *FINANCE) == ([], 6)
    assert problems(form(at(400)), *FINANCE, []) == ([SR.REQUIRED], 7), "a tightening: no judgement of direction"
    assert problems(form(at(800)), *FINANCE, []) == ([SR.REQUIRED], 7)
    # leaving protection, with nothing enforced changed
    assert problems(form(area="Operations"), [("Operations", False)], [(False,)], []) == ([SR.REQUIRED], 7)
    # reviewed, and the form sent is the reviewed one
    reviewed = [(form(at(400)), "tst-reviewer", None)]
    assert problems(form(at(400)), *FINANCE, reviewed) == ([], 7)
    assert problems(form(at(401)), *FINANCE, reviewed) == ([SR.REQUIRED], 7)


def test_the_module_reads_no_decision_and_imports_no_evaluator():
    import inspect
    source = inspect.getsource(SR)
    assert "bp_policy_firing" not in source.split('"""', 2)[2], "outside the docstring, the firing log is not named"
    for name in ("conditions", "enforcement", "learning_patterns", "gates", "loosening"):
        assert not hasattr(SR, name), name


def test_a_policy_that_was_never_live_is_not_covered(no_history):
    cur = _Scripted([[(7,)], [(None,)]])                     # a suggestion row, and no live version ever
    assert SR.activation_problems(cur, "FIN-0001", None, 1, form(at(800))) == []
    assert len(cur.asked) == 2


def test_the_policy_page_says_required_when_it_cannot_tell(caplog):
    versions = [{"version": 2, "savedAs": "live", "form": form()},
                {"version": 3, "savedAs": "draft", "form": form(at(800))}]
    head = {"liveVersion": 2, "latestVersion": 3}
    with caplog.at_level(logging.ERROR, logger=SR.__name__):
        got = SR.status(_Broken(), "FIN-0001", head, versions)
    assert got == {"required": True, "version": 3, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None,
                   "whoMayReview": SR.WHO_MAY_REVIEW}       # stage 5b: who may record it
    assert any("FIN-0001" in r.getMessage() for r in caplog.records)
    # a draft on a retired policy is judged too (against what was last live)
    retired = versions[:1] + [{"version": 3, "savedAs": "retired", "form": form()},
                              {"version": 4, "savedAs": "draft", "form": form(at(800))}]
    assert SR.status(_Broken(), "FIN-0001", {"liveVersion": None, "latestVersion": 4}, retired)["required"] is True


def test_the_policy_page_invites_no_review_when_the_saved_form_cannot_be_read(caplog):
    """The state in which activation answers `second_review_unreadable` and `record` refuses."""
    versions = [{"version": 2, "savedAs": "live", "form": form()},
                {"version": 3, "savedAs": "draft", "form": form(at(800))}]
    head = {"liveVersion": 2, "latestVersion": 3}
    for stored in (("{not json", None), ("[1, 2]", None), None):          # bad JSON, not an object, no row
        script = [[(7,)], [stored] if stored else []]
        assert SR.activation_problems(_Scripted(script), "FIN-0001", 2, 3, form(at(800))) == [SR.UNREADABLE]
        with caplog.at_level(logging.ERROR, logger=SR.__name__):
            got = SR.status(_Scripted(script), "FIN-0001", head, versions)
        assert got == {"required": True, "version": 3, "reviewedBy": None, "reviewedByName": None,
                       "reviewedAt": None}, stored
    assert any("FIN-0001" in r.getMessage() for r in caplog.records)


def test_the_policy_page_asks_nothing_when_there_is_no_draft_on_a_live_version():
    none = {"required": False, "version": None, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}
    live = {"version": 2, "savedAs": "live", "form": form()}
    draft = {"version": 1, "savedAs": "draft", "form": form()}
    retired = {"version": 3, "savedAs": "retired", "form": form()}
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
    ("identity_incomplete", "Your sign-in carries no email address, so we cannot tell whether you are this "
                            "policy's owner."),
    ("second_review_unavailable", "Can't check whether a second review is needed right now. Try again shortly."),
    ("second_review_unreadable", "This policy's saved form could not be read, so a second review cannot be checked. "
                                 "An administrator needs to look at it."),
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
    assert SR.MESSAGE == "A second reviewer must review this change before it goes Active."
    assert SR.REFUSALS == {
        "own_change": "You saved this change, so someone else must review it.",
        "owner": "The policy's owner cannot be its second reviewer.",
        "owner_unmapped": "Nobody is linked to this policy's owner yet, so a second review cannot be recorded. "
                          "An administrator must link them first.",
        "not_latest": "This is no longer the latest draft of the policy. Reload and try again.",
        "not_needed": "This change does not need a second review.",
        "already_reviewed": "This draft has already been reviewed.",
        "identity_incomplete": "Your sign-in carries no email address, so we cannot tell whether you are this "
                               "policy's owner.",
        "second_review_unavailable": "Can't check whether a second review is needed right now. Try again shortly.",
        "second_review_unreadable": "This policy's saved form could not be read, so a second review cannot be "
                                    "checked. An administrator needs to look at it.",
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
