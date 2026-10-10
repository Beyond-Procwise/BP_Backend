"""The second reviewer, without a database: the rule (which changes need one), and how the
endpoint answers.

Numbers as everywhere in stage 5: the live policy is "args.amount gt 500" on one tool, and people
decided on amounts 510, 520, ... 800 (the witnesses). A change needs the review when it stops
asking a person about an action one of them shows; 400 is a tightening.
"""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import conditions, enforcement
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from services.agent_policy import settings as S

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


def wit(value=510, *, tool="refund.issue", missing=(), **more):
    """A firing row as the gate writes it for the live policy: the fields its condition read."""
    found = {"tool.name": tool, **more}
    if value is not None:
        found["args.amount"] = value
    return {"firing_id": 1, "checkpoint": "tool.call.before", "action_name": tool, "matched_values": found,
            "missing_inputs": list(missing)}


WITNESSES = [wit(a) for a in range(510, 801, 10)]


def looser(new, witnesses=WITNESSES, live=None):
    return SR.loosening(live or form(), new, witnesses, SETTINGS)


# ------------------------------------------------------------------ gates (pure)
def test_gates_says_whether_the_action_would_still_pause_for_a_person():
    assert SR.gates(form(), wit(510), SETTINGS) is True
    assert SR.gates(form(at(800)), wit(510), SETTINGS) is False
    assert SR.gates(form(at(800)), wit(810), SETTINGS) is True
    assert SR.gates(form(outcome="notify"), wit(510), SETTINGS) is False
    assert SR.gates(form(outcome="block"), wit(510), SETTINGS) is False, "blocked is not 'a person decides'"
    assert SR.gates(form(checkpoint="agent.reply.before"), wit(510), SETTINGS) is False


def test_gates_reads_the_tool_from_the_action_name_as_enforcement_does():
    assert SR.gates(form(tools=["credit.issue"]), wit(510), SETTINGS) is False, "the policy no longer lists the tool"
    assert SR.gates(form(tools=["credit.issue", "refund.issue"]), wit(510), SETTINGS) is True
    assert SR.gates(form(tools=[]), wit(510), SETTINGS) is True, "no action listed: the policy covers every tool"
    # a row of a version whose condition never read tool.name still knows its tool
    bare = {"firing_id": 2, "checkpoint": "tool.call.before", "action_name": "refund.issue",
            "matched_values": {"args.amount": 510}, "missing_inputs": []}
    assert SR.gates(form(), bare, SETTINGS) is True
    assert SR.gates(form({"all": [{"field": "tool.name", "op": "in", "value": ["credit.issue"]}, amount("gt", 500)]}),
                    bare, SETTINGS) is False
    # and the column wins over a value that disagrees with it
    odd = {**bare, "matched_values": {"tool.name": "something.else", "args.amount": 510}}
    assert SR.gates(form(), odd, SETTINGS) is True


def test_gates_cannot_tell_about_a_field_the_decision_never_recorded():
    reads_currency = {"all": [TOOL, amount("gt", 500), {"field": "args.currency", "op": "eq", "value": "ZZZ"}]}
    assert SR.gates(form(reads_currency), wit(510), SETTINGS) is None, "not recorded is not the same as absent"
    assert SR.gates(form(reads_currency), wit(510, **{"args.currency": "USD"}), SETTINGS) is False
    assert SR.gates(form(reads_currency), wit(510, **{"args.currency": "ZZZ"}), SETTINGS) is True
    either = {"any": [LIVE, {"field": "args.currency", "op": "eq", "value": "ZZZ"}]}
    assert SR.gates(form(either), wit(510), SETTINGS) is None, "even where the known part settles it: fail closed"


def test_a_field_the_action_really_lacked_is_judged_by_the_forms_missing_data_rule():
    lacked = wit(None, missing=["args.amount"])
    assert SR.gates(form(), lacked, SETTINGS) is True, "missing data fails closed: a person was asked"
    assert SR.gates(form(on_missing="fail_open"), lacked, SETTINGS) is False
    assert SR.gates(form(on_missing="anything else"), lacked, SETTINGS) is True, "only an explicit fail_open opens"
    company = S.merge({"on_missing_data": {"approve": "fail_open"}})
    assert SR.gates(form(), lacked, company) is False, "the company setting, when the form names none"
    exists = {"all": [TOOL, {"field": "args.amount", "op": "exists"}]}
    assert SR.gates(form(exists), lacked, SETTINGS) is False and SR.gates(form(exists), wit(510), SETTINGS) is True
    # a null recorded as found is absent, as the evaluator reads it
    assert SR.gates(form(), {**wit(None), "matched_values": {"args.amount": None}}, SETTINGS) is True


@pytest.mark.parametrize("witness", [
    wit(enforcement.MASK),                                          # the value is hidden
    {**wit(510), "matched_values": None},                           # the row cannot be read
    {**wit(510), "missing_inputs": None},
    {"firing_id": 3},
])
def test_gates_cannot_tell_from_a_decision_it_cannot_read(witness):
    assert SR.gates(form(), witness, SETTINGS) is None


@pytest.mark.parametrize("bad", [None, {}, {"outcome": "approve"}, form(None), form({"any": []}),
                                 form({"field": "args.amount", "op": "about", "value": 5}),
                                 form(amount("gt", "a lot")),
                                 {**form(), "hidden": {**form()["hidden"], "actions": {"tools": "refund.issue"}}}])
def test_gates_cannot_tell_from_a_form_it_cannot_read(bad):
    assert SR.gates(bad, wit(510), SETTINGS) is None


def test_gates_agrees_with_enforcement_on_the_action_the_decision_was_about():
    """The same form, compiled the way a save compiles the parts enforcement reads, checked by
    enforcement.check on the action the witness records."""
    def enforced(f, w):
        policy = {"id": "P", "version": 1, "context": {"checkpoint": f["hidden"]["checkpoint"],
                                                       "actions": f["hidden"]["actions"]},
                  "trigger": {"condition": f["hidden"]["condition"],
                              "onMissingData": SR.on_missing_for(f, SETTINGS)},
                  "enforcement": {"outcome": f["outcome"]}}
        ctx = {"checkpoint": w["checkpoint"], **w["matched_values"], "tool.name": w["action_name"]}
        return enforcement.check(ctx, [policy]).result == "paused_for_approval"

    shapes = [LIVE, at(800), at(499), {"any": [at(800)]}, {"all": [TOOL, {"not": amount("lte", 800)}]},
              {"all": [TOOL, amount("gt", 500), amount("gte", 800)]}]
    forms = [form(c) for c in shapes] + [form(outcome="notify"), form(tools=["credit.issue"]), form(tools=[]),
                                         form(on_missing="fail_open")]
    for f in forms:
        for w in (wit(510), wit(800), wit(810), wit(500), wit(None, missing=["args.amount"]),
                  wit(510, tool="credit.issue")):
            assert SR.gates(f, w, SETTINGS) is enforced(f, w), (f, w)


# ------------------------------------------------------------------ the rule (pure)
SHAPES = {   # every way the review of "500 -> 800" was got round, and their kin
    "gt 800": at(800),
    "gte 800": {"all": [TOOL, amount("gte", 800)]},
    "gt 500 AND gte 800": {"all": [TOOL, amount("gt", 500), amount("gte", 800)]},
    "gt 800 listed twice": {"all": [TOOL, amount("gt", 800), amount("gt", 800)]},
    "any[gt 800] inside the all": {"all": [TOOL, {"any": [amount("gt", 800)]}]},
    "not lte 800": {"all": [TOOL, {"not": amount("lte", 800)}]},
    "a top-level any wrapper": {"any": [at(800)]},
    "a smaller loosening": at(600),
    "beyond what was suggested": at(2000),
    "the tool swapped in the leaf": {"all": [{"field": "tool.name", "op": "in", "value": ["credit.issue"]},
                                             amount("gt", 500)]},
    "a new field read (cannot tell)": {"all": [TOOL, amount("gt", 500),
                                               {"field": "args.currency", "op": "eq", "value": "ZZZ"}]},
    "not a condition at all": None,
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_shape_that_lets_a_decided_action_go_is_looser(name):
    assert conditions.to_engine(SHAPES[name]) if SHAPES[name] else True      # shapes the real evaluator accepts
    assert looser(form(SHAPES[name])) is True
    # and each is let go by a particular decision: the one at 510
    assert SR.gates(form(), wit(510), SETTINGS) is True and SR.gates(form(SHAPES[name]), wit(510), SETTINGS) is not True


def test_what_keeps_asking_about_every_decision_is_not_looser():
    for name, new in {
        "unchanged": form(),
        "a tightening": form(at(400)),
        "the same limit, written another way": form({"all": [TOOL, {"not": amount("lte", 500)}]}),
        "another tool added": form({"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                                            amount("gt", 500)]}, tools=["refund.issue", "credit.issue"]),
        "no action listed (every tool)": form(tools=[]),
        "fail closed said out loud": form(on_missing="fail_closed"),
    }.items():
        assert looser(new) is False, name
    assert looser(form(at(800)), witnesses=[]) is False, "nobody ever decided anything: no evidence (a known limit)"
    assert looser(form(at(800)), witnesses=[wit(810), wit(900)]) is False, "no decision in the region let go"


def test_changes_that_touch_no_condition_are_looser_too():
    assert looser(form(outcome="notify")) is True
    assert looser(form(tools=["credit.issue"])) is True
    assert looser(form(checkpoint="agent.reply.before")) is True
    two = form(tools=["refund.issue", "credit.issue"])
    assert looser(form(tools=["credit.issue"]), live=two) is True, "one of several tools removed"


def test_fail_open_is_looser_whatever_the_decisions_show():
    # ruling A: every decision carried its amount, so the evidence is silent
    assert all(SR.gates(form(on_missing="fail_open"), w, SETTINGS) is True for w in WITNESSES)
    assert looser(form(on_missing="fail_open")) is True
    assert looser(form(on_missing="fail_open"), witnesses=[]) is True
    already = form(on_missing="fail_open")
    assert SR.loosening(already, form(on_missing="fail_open"), WITNESSES, SETTINGS) is False, "it already was"
    assert SR.loosening(already, form(), WITNESSES, SETTINGS) is False, "and closing it is a tightening"


def test_a_decision_that_cannot_be_read_demands_the_review_of_any_change_but_an_identical_one():
    # the row exists because some version of the policy paused that action: "cannot tell" does not excuse
    hidden = [wit(enforcement.MASK), {**wit(510), "matched_values": None}]
    for unreadable in hidden:
        assert looser(form(at(800)), witnesses=[unreadable]) is True
        assert looser(form(at(400)), witnesses=[unreadable] + WITNESSES) is True, "even a tightening: cannot tell"
        assert looser(form(), witnesses=[unreadable] + WITNESSES) is False, "nothing enforced is changed"
    assert looser(form(at(800)), witnesses=hidden + WITNESSES) is True
    # a decision the live form provably does not gate (another tool) says nothing
    assert looser(form(at(400)), witnesses=[wit(enforcement.MASK, tool="credit.issue")] + WITNESSES) is False


# ------------------------------------------------------------------ missing data, asked directly (fix round 2, finding 1)
EXISTS = {"field": "args.amount", "op": "exists"}
GUARDS = {
    "an exists guard": {"all": [TOOL, EXISTS, amount("gt", 500)]},
    "the guard written last": {"all": [TOOL, amount("gt", 500), EXISTS]},
    "not(not exists)": {"all": [TOOL, {"not": {"not": EXISTS}}, amount("gt", 500)]},
}


def _result(f, ctx):
    """What enforcement answers for one action under the form, compiled the way a save compiles it."""
    policy = {"id": "P", "version": 1, "context": {"checkpoint": f["hidden"]["checkpoint"],
                                                   "actions": f["hidden"]["actions"]},
              "trigger": {"condition": f["hidden"]["condition"], "onMissingData": SR.on_missing_for(f, SETTINGS)},
              "enforcement": {"outcome": f["outcome"]}}
    return enforcement.check({"checkpoint": "tool.call.before", **ctx}, [policy]).result


@pytest.mark.parametrize("name", sorted(GUARDS))
def test_a_guard_that_lets_an_action_without_the_value_through_needs_the_review(name):
    """The reviewer's example. Every decision carried its amount, so no real one shows it; the
    question is asked of the newest decision with the value taken away."""
    new = form(GUARDS[name])
    assert conditions.to_engine(GUARDS[name])
    no_amount = {"tool.name": "refund.issue"}
    assert (_result(form(), no_amount), _result(new, no_amount)) == ("paused_for_approval", "allowed")
    assert all(SR.gates(new, w, SETTINGS) is True for w in WITNESSES), "the real decisions are silent"
    assert looser(new) is True
    assert looser(new, witnesses=[wit(900)]) is True, "one decision is enough to ask it of"
    assert looser(new, witnesses=[]) is False, "nobody ever decided anything: a known limit"


def test_a_change_that_leaves_missing_data_alone_needs_no_review():
    reworded = {**form(), "messageForAgent": "reworded", "owner": "Someone else"}
    for name, new in {"an unrelated wording edit": reworded, "a tightening": form(at(400)),
                      "resubmitted unchanged": form(),
                      "a guard that changes nothing": form({"all": [TOOL, amount("gt", 500), {"not": amount("lte", 500)}]}),
                      }.items():
        assert looser(new) is False, name
        assert _result(new, {"tool.name": "refund.issue"}) == "paused_for_approval", name


def test_the_missing_data_question_is_asked_of_every_field_the_live_condition_reads():
    country = {"field": "customer.country", "op": "in", "value": ["FR", "DE"]}
    live = form({"all": [TOOL, amount("gt", 500), country]})
    seen = [wit(a, **{"customer.country": "FR"}) for a in (510, 520)]
    assert looser(live, witnesses=seen, live=live) is False
    assert looser(form({"all": [TOOL, amount("gt", 400), country]}), witnesses=seen, live=live) is False
    # either field guarded: an action without it stops being asked about
    for guard in ({"field": "customer.country", "op": "exists"}, EXISTS):
        assert looser(form({"all": [TOOL, amount("gt", 500), country, guard]}), witnesses=seen, live=live) is True
    # only an action with NEITHER is let go: the witness with every field taken away shows it
    neither = {"any": [EXISTS, {"field": "customer.country", "op": "exists"}]}
    assert looser(form({"all": [TOOL, amount("gt", 500), country, neither]}), witnesses=seen, live=live) is True
    # a live form that already lets missing data through has nothing to lose
    open_live = form({"all": [TOOL, amount("gt", 500)]}, on_missing="fail_open")
    assert SR.loosening(open_live, form(GUARDS["an exists guard"], on_missing="fail_open"), WITNESSES, SETTINGS) is False


# ------------------------------------------------------------------ a policy that reads a hidden value (finding 2)
VENDOR = {"field": "args.vendor", "op": "eq", "value": "X"}
MASKED_LIVE = {"all": [TOOL, amount("gt", 500), VENDOR]}
MASKED = [wit(a, **{"args.vendor": enforcement.MASK}) for a in range(510, 801, 10)]


def test_a_policy_that_reads_a_hidden_value_is_covered():
    """The reviewer's example: every decision has the vendor stored as the mask, so the live form
    cannot be SHOWN to gate any of them. They still count: each row is there because the policy
    paused that action."""
    live = form(MASKED_LIVE)
    assert {SR.gates(live, w, SETTINGS) for w in MASKED} == {None}
    assert looser(form({"all": [TOOL, amount("gt", 800), VENDOR]}), witnesses=MASKED, live=live) is True
    assert looser(form(at(800)), witnesses=MASKED, live=live) is True, "the hidden leaf dropped as well"
    # whichever decision is the newest: here it is one the new limit still asks about, so the
    # missing-data question finds nothing and only the real decisions can show the loosening
    newest_above = [wit(900, **{"args.vendor": enforcement.MASK})] + MASKED
    assert looser(form({"all": [TOOL, amount("gt", 800), VENDOR]}), witnesses=newest_above, live=live) is True


def test_an_edit_that_changes_nothing_enforced_needs_no_review_even_when_nothing_can_be_told():
    live = form(MASKED_LIVE)
    reworded = {**form(MASKED_LIVE), "owner": "Another owner", "messageForAgent": "reworded", "deciders": ["X"]}
    assert looser(reworded, witnesses=MASKED, live=live) is False
    assert looser(form(MASKED_LIVE), witnesses=MASKED, live=live) is False, "resubmitted unchanged"
    # each part of what is enforced, changed alone, is no longer "the same"
    for name, new in {"outcome": form(MASKED_LIVE, outcome="block"),
                      "checkpoint": form(MASKED_LIVE, checkpoint="agent.reply.before"),
                      "tools": form(MASKED_LIVE, tools=["refund.issue", "credit.issue"]),
                      "condition": form({"all": [TOOL, amount("gt", 500.0001), VENDOR]}),
                      "missing data": form(MASKED_LIVE, on_missing="fail_open")}.items():
        assert looser(new, witnesses=MASKED, live=live) is True, name
    # the missing-data rule is compared as enforced, not as typed: the company's rule said out loud is the same
    assert looser(form(MASKED_LIVE, on_missing="fail_closed"), witnesses=MASKED, live=live) is False


def test_a_tightening_of_a_policy_that_reads_a_hidden_value_needs_the_review():
    # an accepted cost: it cannot be told apart from a loosening
    assert looser(form({"all": [TOOL, amount("gt", 400), VENDOR]}), witnesses=MASKED, live=form(MASKED_LIVE)) is True


def test_the_two_step_route_is_measured_against_decisions_made_under_any_version():
    # live is now "gt 499" (a tightening activated first); the decisions were made under "gt 500"
    assert looser(form(at(800)), live=form(at(499))) is True
    assert looser(form(at(800)), live=form({"all": [TOOL, amount("gte", 500.01)]})) is True


def test_a_review_covers_the_whole_form_but_for_the_confirmation_and_the_note():
    stored = {**form(at(800)), "checked": {"by": "a", "at": "t"}, "changeNote": "one", "deciders": ["CFO"]}
    assert SR._same_form(stored, {**stored, "checked": None, "changeNote": "two"})
    assert SR._same_form(stored, json.loads(json.dumps(stored)))
    for key, value in (("outcome", "notify"), ("deciders", ["Someone"]), ("owner", "Other"),
                       ("businessArea", "Operations"), ("messageForAgent", "reworded")):
        assert not SR._same_form(stored, {**stored, key: value}), key
    for key, value in (("onMissingData", "fail_open"), ("actions", {"tools": ["credit.issue"]}), ("condition", at(900))):
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
    """The decision history and the settings must not be read at all."""
    def boom(*a, **k):
        raise AssertionError("not for this policy")
    monkeypatch.setattr(SR.learning, "witnesses", boom)
    monkeypatch.setattr(SR, "load_settings", boom)


def test_a_policy_with_no_suggestion_costs_one_select_and_needs_nothing(no_history):
    for live in (2, None):                                   # live now, or never live / retired
        cur = _Scripted([[]])
        assert SR.activation_problems(cur, "OPS-0001", live, 2, form(at(800))) == []
        assert len(cur.asked) == 1 and "bp_agent_policy_suggestion" in cur.asked[0] and "LIMIT 1" in cur.asked[0]


def test_a_policy_outside_a_protected_area_never_has_its_decisions_read(no_history):
    ops = form(area="Operations")
    # the suggestion row, the live form, the area as a save resolves it, its setting
    cur = _Scripted([[(7,)], [(ops,)], [("Operations", False)], [(False,)]])
    assert SR.activation_problems(cur, "OPS-0001", 2, 2, form(at(800), area="Operations")) == []
    assert len(cur.asked) == 4 and not any("bp_policy_firing" in q for q in cur.asked)


FINANCE = [("Finance", False)], [(True,)]          # the area as a save resolves it, then its setting


def _compiled(on_missing):
    return {"trigger": {"onMissingData": on_missing}}


def test_an_edit_that_changes_nothing_enforced_never_has_the_decisions_read(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the decision history must not be read for this change")
    monkeypatch.setattr(SR.learning, "witnesses", boom)
    monkeypatch.setattr(SR, "load_settings", lambda conn: SETTINGS)
    reworded = {**form(MASKED_LIVE), "messageForAgent": "reworded"}
    cur = _Scripted([[(7,)], [(form(MASKED_LIVE), _compiled("fail_closed"))], *FINANCE, *FINANCE])
    cur.connection = None
    assert SR.activation_problems(cur, "FIN-0001", 2, 2, reworded) == []
    assert len(cur.asked) == 6 and not any("bp_policy_firing" in q for q in cur.asked)
    # but it is still asked where the policy is going: out of the protected area needs the review
    cur = _Scripted([[(7,)], [(form(MASKED_LIVE), _compiled("fail_closed"))], *FINANCE,
                     [("Operations", False)], [(False,)], []])
    cur.connection = None
    assert SR.activation_problems(cur, "FIN-0001", 2, 2, {**reworded, "businessArea": "Operations"}) == [SR.REQUIRED]


def test_the_live_missing_data_rule_is_the_one_compiled_into_the_live_version(monkeypatch):
    """The live form names no rule and was compiled fail-closed; an Admin has since made the
    company default fail-open. Saving the same form again would compile it fail-open."""
    company = S.merge({"on_missing_data": {"approve": "fail_open"}})
    monkeypatch.setattr(SR, "load_settings", lambda conn: company)
    monkeypatch.setattr(SR.learning, "witnesses", lambda cur, key: list(WITNESSES))

    def problems(compiled, new=None):
        cur = _Scripted([[(7,)], [(form(), compiled)], *FINANCE, *FINANCE, []])
        cur.connection = None
        return SR.activation_problems(cur, "FIN-0001", 2, 2, new or {**form(), "messageForAgent": "reworded"})

    assert problems(_compiled("fail_closed")) == [SR.REQUIRED], "missing data would newly be let through"
    assert problems(_compiled("fail_closed"), form(on_missing="fail_closed")) == [], "kept closed, said out loud"
    assert problems(_compiled("fail_open")) == [], "it already was open"
    # no compiled document to read: the form and today's settings are all there is
    for unreadable in (None, {}, {"trigger": {}}, {"trigger": {"onMissingData": None}}, "not a document"):
        assert problems(unreadable) == [], repr(unreadable)
    assert problems(json.dumps(_compiled("fail_closed"))) == [SR.REQUIRED], "the document as stored text"


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
    assert got == {"required": True, "version": 3, "reviewedBy": None, "reviewedAt": None}
    assert any("FIN-0001" in r.getMessage() for r in caplog.records)
    # a draft on a retired policy is judged too (against what was last live)
    retired = versions[:1] + [{"version": 3, "savedAs": "retired", "form": form()},
                              {"version": 4, "savedAs": "draft", "form": form(at(800))}]
    assert SR.status(_Broken(), "FIN-0001", {"liveVersion": None, "latestVersion": 4}, retired)["required"] is True


def test_the_policy_page_asks_nothing_when_there_is_no_draft_on_a_live_version():
    none = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}
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
