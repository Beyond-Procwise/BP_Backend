"""The second reviewer on bp_testdb, through the real save path. Needs PROCWISE_TEST_LIVE_DB=1.

Each test has REAL live policies ("TST …", on a tool named tst_<tag>, RETIRED at teardown): one in
Finance (second reviewer on, by seed) and, where it says so, one in Operations (off). Each gets
the standard evidence and its one suggestion from the real scan (worked in
test_learning_scan_live's docstring): 33 approvals at 510..830 under "args.amount gt 500" ->
the suggestion is 500 -> 800. The firing rows are written as the gate writes them (the policy's
own tool in `action_name` and in `tool.name`), because they are the WITNESSES the rule reads: a
change needs the second review when it stops asking a person about an action one of them shows.
So 800, 600 and 2000 are loosenings (510 is no longer asked about), 400 is a tightening.

readiness and the final check are bypassed (tst_ tools are unknown to the registry); every
restructured condition is therefore also put through the real conditions.to_engine (`_ok`).

Review rows, suggestion rows and the decider-map row a test makes are removed by the fixture;
firing rows are append-only BY DESIGN and stay.
"""
import copy
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.auth import Principal
from api.routers import agent_policies as R
from repositories import agent_policy_repo as repo
from services.agent_policy import conditions
from services.agent_policy import learning as L
from services.agent_policy import second_review as SR
from tests.agent_policy.test_learning_endpoints_live import _form
from services.agent_policy.enforcement import MASK
from tests.agent_policy.test_learning_scan_live import BASE, NOW, fire

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REQUIRED = [{"field": "secondReview", "code": "second_review_required", "message": SR.MESSAGE}]
UNAVAILABLE = [{"field": "secondReview", "code": "second_review_unavailable",
                "message": "Can't check whether a second review is needed right now. Try again shortly."}]
NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}
ROLES = {"tst-buyer": "Buyer", "tst-saver": "Approver", "tst-reviewer": "Approver"}


def person(sub, email=None):
    return Principal(subject=sub, email=email or f"{sub}@example.test", claims={"cognito:groups": []})


SAVER, REVIEWER = person("tst-saver"), person("tst-reviewer")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "second-review tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def make(conn, monkeypatch):
    """make(area) -> a live approve policy "amount over 500" in that business area, with the base
    evidence, its one open suggestion, and a person linked to its owner."""
    cur = conn.cursor()
    cur.execute("SELECT area_name, second_reviewer FROM proc.bp_business_area WHERE area_name IN ('Finance','Operations')")
    assert dict(cur.fetchall()) == {"Finance": True, "Operations": False}, \
        "these tests need the seeded settings: second reviewer on for Finance, off for Operations"
    # tst_ tools are unknown to the real registry: readiness and the final check are bypassed.
    # The second-review check is NOT one of them: it runs for real in every save below.
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    made = []

    def _make(area="Finance", countries=False):
        tag = uuid.uuid4().hex[:8]
        w = SimpleNamespace(tag=tag, key=None, live=2, owner=f"TST Owner {tag}", tool=f"tst_{tag}",
                            owner_person=person(f"tst-owner-{tag}"))
        made.append(w)
        form = _form(tag)
        form["businessArea"] = area
        extra = None
        if countries:   # a second limit, a list: 30 decisions on FR, 3 on DE -> "stop asking about FR"
            form["hidden"]["condition"]["all"].append({"field": "customer.country", "op": "in", "value": ["FR", "DE"]})
            form["hidden"]["inputs"].append({"name": "Customer country", "field": "customer.country", "type": "string"})
            extra = [{"customer.country": "FR" if i < 30 else "DE"} for i in range(33)]
        w.key = repo.create_draft(conn, form, actor="test")["policyKey"]
        repo.save_version(conn, w.key, form, base_version=1, intent="activate", actor="test", change_note="go")
        conn.cursor().execute(
            "INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) VALUES (%s, %s, 'test')",
            (w.owner, [w.owner_person.email]))
        fire(conn, w.key, BASE, version=w.live, action=w.tool, extra=extra)
        [target] = [t for t in L.targets(conn.cursor()) if t.key == w.key]
        counts = L.scan(conn, NOW, only=[target])
        assert (counts["raised"], counts["errors"]) == (2 if countries else 1, 0), counts
        w.sid = min(c["id"] for c in L.for_policy(conn.cursor(), w.key))
        return w

    yield _make
    c = conn.cursor()
    for w in made:
        c.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name LIKE %s", (f"TST % {w.tag}",))
        if w.key:
            c.execute("DELETE FROM proc.bp_agent_policy_second_review WHERE policy_key = %s", (w.key,))
            c.execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (w.key,))
            got = repo.get_policy(conn, w.key)
            if got["status"] != "retired":
                repo.retire(conn, w.key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


@pytest.fixture
def fin(make):
    return make("Finance")


def _at(conn, w, limit=None, **changes):
    """The live form with the limit (and anything else) changed."""
    form = copy.deepcopy(repo.get_policy(conn, w.key)["versions"][w.live - 1]["form"])
    if limit is not None:
        form["hidden"]["condition"]["all"][1]["value"] = limit
    form.update(changes)
    return form


def _cond(conn, w, *rest, top=None, **hidden):
    """The live form with its condition rebuilt: the tool leaf AND `rest`, or `top(that)`; and any
    other `hidden` key replaced. The condition is one the real evaluator accepts."""
    form = _at(conn, w)
    if rest or top:
        condition = {"all": [form["hidden"]["condition"]["all"][0], *rest]}
        form["hidden"]["condition"] = top(condition) if top else condition
    form["hidden"].update(hidden)
    conditions.to_engine(form["hidden"]["condition"])
    return form


def amount(op, value):
    return {"field": "args.amount", "op": op, "value": value}


def _tools(conn, w, form, tools):
    form["hidden"]["actions"] = {**form["hidden"]["actions"], "tools": list(tools)}
    return form


def _draft(conn, w, form, *, base, actor="tst-saver"):
    return repo.save_version(conn, w.key, form, base_version=base, intent="draft", actor=actor,
                             change_note="TST draft")["version"]


def _activate(conn, w, form, *, base, actor="tst-saver"):
    return repo.save_version(conn, w.key, form, base_version=base, intent="activate", actor=actor,
                             change_note="TST activate")["version"]


def _refused(conn, w, form, *, base, actor="tst-saver"):
    before = _policy(conn, w)
    with pytest.raises(repo.NotReady) as exc:
        _activate(conn, w, form, base=base, actor=actor)
    assert _policy(conn, w) == before, "nothing written, the pointers unmoved"
    return exc.value.problems


def _policy(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT (SELECT count(*) FROM proc.bp_agent_policy_version WHERE policy_key = %s), status, "
                "live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s", (w.key, w.key))
    return cur.fetchone()


def _suggestion(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT status FROM proc.bp_agent_policy_suggestion WHERE suggestion_id = %s", (w.sid,))
    return cur.fetchone()[0]


def _reviews(conn, w):
    cur = conn.cursor()
    cur.execute("SELECT version, suggestion_id, reviewed_by, reviewed_at, note FROM "
                "proc.bp_agent_policy_second_review WHERE policy_key = %s ORDER BY version", (w.key,))
    return cur.fetchall()


def _review(conn, w, version, who=REVIEWER, note=None):
    return SR.record(conn, w.key, version, principal=who, note=note, now=NOW)


def _code(conn, w, version, who):
    with pytest.raises(L.Refused) as exc:
        _review(conn, w, version, who)
    assert exc.value.message == SR.REFUSALS[exc.value.code]
    return exc.value.code


# ------------------------------------------------------------------ activation
def test_finance_activation_of_the_suggested_loosening_is_refused_without_a_review(conn, fin):
    form = _at(conn, fin, 800)
    # straight from the live version, with no draft saved: the suggestion is still open
    assert _refused(conn, fin, form, base=fin.live) == REQUIRED
    assert _suggestion(conn, fin) == "open", "the refused save marked nothing"
    v = _draft(conn, fin, form, base=fin.live)
    assert v == 3 and _suggestion(conn, fin) == "saved", "saving the loosening marks its suggestion saved"
    assert _refused(conn, fin, form, base=v) == REQUIRED, "a saved suggestion asks for the review just the same"
    assert _policy(conn, fin) == (3, "live", fin.live, 3) and _reviews(conn, fin) == []


def test_after_another_approver_reviews_it_activation_succeeds(conn, fin):
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    assert _review(conn, fin, v, note="  the evidence holds  ") == {
        "policyKey": fin.key, "version": v, "reviewedBy": "tst-reviewer", "reviewedAt": NOW.isoformat()}
    assert _reviews(conn, fin) == [(v, fin.sid, "tst-reviewer", NOW, "the evidence holds")]
    assert _code(conn, fin, v, person("tst-third")) == "already_reviewed"
    assert _activate(conn, fin, form, base=v) == v + 1
    assert _policy(conn, fin) == (4, "live", 4, 4)
    assert _reviews(conn, fin) == [(v, fin.sid, "tst-reviewer", NOW, "the evidence holds")], "the review stays on record"


def test_the_second_reviewer_may_be_the_one_who_activates(conn, fin):
    # the rules are about who reviews, not who presses Activate
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v, actor="tst-reviewer") == v + 1
    assert _policy(conn, fin)[1:] == ("live", v + 1, v + 1)


def test_the_reviewed_form_as_the_policy_page_returns_it_activates_unchanged(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    stored = repo.get_policy(conn, fin.key)["versions"][v - 1]["form"]
    assert "checked" in stored and "changeNote" in stored
    # `checked` and `changeNote` are the two keys that may differ: a fresh tick, another note
    resent = {**copy.deepcopy(stored), "checked": {"by": "someone", "at": "2026-10-10T10:00:00Z"}, "changeNote": "now"}
    assert _activate(conn, fin, resent, base=v) == v + 1


def test_a_review_of_one_draft_does_not_cover_a_different_condition(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    assert _refused(conn, fin, _at(conn, fin, 900), base=v) == REQUIRED, "reviewed at 800, activated at 900"
    assert _refused(conn, fin, _at(conn, fin, 600), base=v) == REQUIRED, "nor a smaller loosening nobody reviewed"
    assert _activate(conn, fin, _at(conn, fin, 800), base=v) == v + 1


def test_a_review_of_one_draft_does_not_cover_the_draft_saved_after_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    newer = _draft(conn, fin, _at(conn, fin, 900), base=v, actor="tst-other-saver")
    assert _refused(conn, fin, _at(conn, fin, 900), base=newer) == REQUIRED
    assert _code(conn, fin, v, person("tst-third")) == "not_latest", "the reviewed draft is no longer the latest"
    _review(conn, fin, newer, person("tst-third"))
    assert _activate(conn, fin, _at(conn, fin, 900), base=newer) == newer + 1


@pytest.mark.parametrize("state,limit", [("dismissed", 800), ("not_yet", 800), ("saved", 800),
                                         ("dismissed", 600), ("dismissed", 2000)])
def test_typing_the_loosening_by_hand_still_needs_the_review(conn, fin, state, limit):
    """Ruling Q3. The owner never used the card: they hid it (or left it) and typed the number.
    (The suggestion still `open` at the moment of activation is the first test above.)"""
    if state in ("dismissed", "not_yet"):
        L.act(conn, fin.sid, {"dismissed": "dismiss", "not_yet": "not_yet"}[state], actor="tst-buyer", now=NOW)
    form = _at(conn, fin, limit)
    v = _draft(conn, fin, form, base=fin.live)
    assert _refused(conn, fin, form, base=v) == REQUIRED
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v) == v + 1


# ------------------------------------------------------------------ the bypass routes of the review (fix round 1)
def _wrap_any(condition):
    return {"any": [condition]}


def test_route_1_a_restructured_condition_is_judged_by_what_it_stops_asking_about(conn, fin):
    shapes = {
        "gte 800": _cond(conn, fin, amount("gte", 800)),
        "gt 500 AND gte 800": _cond(conn, fin, amount("gt", 500), amount("gte", 800)),
        "gt 800 listed twice": _cond(conn, fin, amount("gt", 800), amount("gt", 800)),
        "any[gt 800] inside the all": _cond(conn, fin, {"any": [amount("gt", 800)]}),
        "not lte 800": _cond(conn, fin, {"not": amount("lte", 800)}),
        "a top-level any wrapper": _cond(conn, fin, amount("gt", 800), top=_wrap_any),
    }
    for name, form in shapes.items():
        assert _refused(conn, fin, form, base=fin.live) == REQUIRED, name
    v = _draft(conn, fin, shapes["not lte 800"], base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"]["required"] is True, "the page says the same"
    assert _refused(conn, fin, shapes["not lte 800"], base=v) == REQUIRED


def test_route_1_a_restructured_list_is_judged_the_same_way(conn, make):
    w = make("Finance", countries=True)
    de = {"field": "customer.country", "op": "in", "value": ["DE"]}
    for name, form in {
        "FR dropped, the leaf listed twice": _cond(conn, w, amount("gt", 500), de, de),
        "FR dropped, the leaf wrapped in any": _cond(conn, w, amount("gt", 500), {"any": [de]}),
        "FR dropped": _cond(conn, w, amount("gt", 500), de),
    }.items():
        assert _refused(conn, w, form, base=w.live) == REQUIRED, name
    # DE and FR both still asked about, and one more country added: nothing a person decided is let go
    wider = {"field": "customer.country", "op": "in", "value": ["FR", "DE", "IT"]}
    assert _activate(conn, w, _cond(conn, w, amount("gt", 500), wider), base=w.live) == 3


def test_route_2_a_tightening_first_does_not_reset_what_the_loosening_is_measured_against(conn, fin):
    assert _activate(conn, fin, _at(conn, fin, 499), base=2) == 3, "a tightening needs no review"
    fin.live = 3
    assert _refused(conn, fin, _at(conn, fin, 800), base=3) == REQUIRED, "the decisions were made under version 2"
    assert _activate(conn, fin, _cond(conn, fin, amount("gte", 500.01)), base=3) == 4, "every decision is still asked about"
    fin.live = 4
    assert _refused(conn, fin, _cond(conn, fin, amount("gt", 800)), base=4) == REQUIRED


def test_route_2_a_superseded_suggestion_still_counts(conn, fin):
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'superseded' WHERE suggestion_id = %s",
                          (fin.sid,))
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED


def test_route_3_moving_the_policy_out_of_the_protected_area_needs_the_review_itself(conn, fin):
    moved = _at(conn, fin, businessArea="Operations")           # the condition untouched
    assert _refused(conn, fin, moved, base=fin.live) == REQUIRED
    v = _draft(conn, fin, moved, base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    assert _refused(conn, fin, moved, base=v) == REQUIRED
    assert _refused(conn, fin, _at(conn, fin, businessArea="TST no such area"), base=v) == REQUIRED, \
        "an unknown area resolves, as a save resolves it, to the unassigned one"
    assert _refused(conn, fin, _at(conn, fin, 800, businessArea="Operations"), base=v) == REQUIRED


def test_route_4_a_review_covers_the_whole_reviewed_form_not_only_its_condition(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    _review(conn, fin, v)
    other = f"TST Other Owner {fin.tag}"
    changed = {
        "outcome": _at(conn, fin, 800, outcome="notify", notify=["TST Someone"]),
        "deciders": _at(conn, fin, 800, deciders=["TST Someone Else"]),
        "onMissingData": _cond(conn, fin, amount("gt", 800), onMissingData="fail_open"),
        "tools": _tools(conn, fin, _at(conn, fin, 800), [fin.tool, "tst_other"]),
        "owner": _at(conn, fin, 800, owner=other),
        "message": _at(conn, fin, 800, messageForAgent="TST reworded"),
    }
    for name, form in changed.items():
        assert _refused(conn, fin, form, base=v) == REQUIRED, name
    assert _activate(conn, fin, _at(conn, fin, 800), base=v) == v + 1


def test_route_5_loosening_without_touching_the_condition(conn, fin):
    live = _at(conn, fin)
    assert _refused(conn, fin, _at(conn, fin, outcome="notify", notify=["TST Someone"]), base=2) == REQUIRED, "outcome"
    assert _refused(conn, fin, _tools(conn, fin, _at(conn, fin), ["tst_other"]), base=2) == REQUIRED, \
        "the tool replaced in the list of actions"
    swapped = _at(conn, fin)
    swapped["hidden"]["condition"]["all"][0] = {"field": "tool.name", "op": "in", "value": ["tst_other"]}
    assert _refused(conn, fin, swapped, base=2) == REQUIRED, "the tool replaced in the tool.name leaf"
    assert _refused(conn, fin, _cond(conn, fin, onMissingData="fail_open"), base=2) == REQUIRED, \
        "fail-open: no decision shows it (every one carried its amount), so this one is not judged by evidence"
    # one of several: first widen the policy to a second tool (nothing is let go: no review) ...
    both = _tools(conn, fin, copy.deepcopy(live), [fin.tool, "tst_other"])
    both["hidden"]["condition"]["all"][0]["value"] = [fin.tool, "tst_other"]
    assert _activate(conn, fin, both, base=2) == 3
    fin.live = 3
    # ... then take away the one people decided on
    assert _refused(conn, fin, _tools(conn, fin, _at(conn, fin), ["tst_other"]), base=3) == REQUIRED


def test_an_emptied_list_of_actions_covers_every_tool_and_needs_no_review(conn, fin):
    # exactly as enforcement reads it: a policy that lists no action applies to all of them
    assert _activate(conn, fin, _tools(conn, fin, _at(conn, fin), []), base=fin.live) == 3


def test_route_6_nobody_who_saved_a_version_since_the_live_one_may_review(conn, fin):
    form = _at(conn, fin, 800)
    v3 = _draft(conn, fin, form, base=fin.live, actor="tst-saver")
    v4 = _draft(conn, fin, form, base=v3, actor="tst-b")
    assert _code(conn, fin, v4, SAVER) == "own_change", "A saved the loosening; B only re-saved it"
    assert _code(conn, fin, v4, person("tst-b")) == "own_change"
    assert _reviews(conn, fin) == []
    _review(conn, fin, v4)


def test_route_7_a_retired_policy_cannot_come_back_looser_without_the_review(conn, fin):
    retired = repo.retire(conn, fin.key, base_version=2, actor="tst-saver", change_note="TST retire")["version"]
    assert _policy(conn, fin) == (3, "retired", None, 3)
    form = _at(conn, fin, 800)
    assert _refused(conn, fin, form, base=retired) == REQUIRED, "measured against the last live version"
    v = _draft(conn, fin, form, base=retired, actor="tst-b")
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    assert _code(conn, fin, v, SAVER) == "own_change", "the person who retired it saved a version since it was live"
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v) == v + 1
    assert _policy(conn, fin) == (5, "live", 5, 5), "a retired policy CAN be made live again (stage 1 behaviour)"


def test_a_retired_policy_comes_back_unchanged_without_a_review(conn, fin):
    retired = repo.retire(conn, fin.key, base_version=2, actor="tst-saver", change_note="TST retire")["version"]
    assert _activate(conn, fin, _at(conn, fin), base=retired) == 4
    assert _policy(conn, fin)[1:] == ("live", 4, 4) and _reviews(conn, fin) == []


# ------------------------------------------------------------------ what a decision does not record (rulings A, B)
def test_a_condition_that_reads_a_field_no_decision_recorded_needs_the_review(conn, fin):
    # a real USD refund of 510 would stop being asked about; the decisions never recorded a currency
    narrowed = _cond(conn, fin, amount("gt", 500), {"field": "args.currency", "op": "eq", "value": "ZZZ"})
    assert _refused(conn, fin, narrowed, base=fin.live) == REQUIRED
    widened = _cond(conn, fin, {"any": [amount("gt", 500), {"field": "args.currency", "op": "eq", "value": "ZZZ"}]})
    assert _refused(conn, fin, widened, base=fin.live) == REQUIRED, "cannot tell is not a pass, even for a widening"


def test_a_field_the_action_really_lacked_is_judged_by_the_missing_data_rule(conn, fin):
    # "the amount must be there": every decision so far carried one, so nothing is let go
    must_exist = _cond(conn, fin, amount("gt", 500), {"field": "args.amount", "op": "exists"})
    assert SR.needed(conn.cursor(), fin.key, fin.live, must_exist) is None
    # one decision on an action that carried NO amount (it paused because missing data fails closed)
    fire(conn, fin.key, [None], version=fin.live, action=fin.tool, missing=["args.amount"], first=40)
    [w] = [w for w in L.witnesses(conn.cursor(), fin.key) if w["missing_inputs"]]
    assert (w["action_name"], w["matched_values"], w["missing_inputs"]) == (fin.tool, {"tool.name": fin.tool},
                                                                            ["args.amount"])
    assert _refused(conn, fin, must_exist, base=fin.live) == REQUIRED, "that action would no longer be asked about"
    assert _activate(conn, fin, _at(conn, fin, 400), base=fin.live) == 3, "a tightening still asks about it"


def test_a_decision_whose_value_is_hidden_blocks_nothing(conn, fin):
    fire(conn, fin.key, [MASK], version=fin.live, action=fin.tool, first=41)
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED
    assert _activate(conn, fin, _at(conn, fin, 400), base=fin.live) == 3


def test_a_stored_suggestion_that_cannot_be_read_changes_nothing(conn, fin):
    # the suggestion's content is not consulted: only that learning has spoken
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET proposal = '{\"kind\": \"threshold\"}'::jsonb "
                          "WHERE suggestion_id = %s", (fin.sid,))
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED
    assert _activate(conn, fin, _at(conn, fin, 400), base=fin.live) == 3


# ------------------------------------------------------------------ what needs no review
def _no_witness_read(monkeypatch):
    def boom(cur, key):
        raise AssertionError("the decision history must not be read for this policy")
    monkeypatch.setattr(SR.learning, "witnesses", boom)


def test_a_policy_with_no_suggestion_needs_no_review_even_when_looser(conn, fin, monkeypatch):
    conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (fin.key,))
    _no_witness_read(monkeypatch)
    assert _activate(conn, fin, _at(conn, fin, 800), base=fin.live) == 3


def test_operations_needs_no_second_review(conn, make, monkeypatch):
    ops = make("Operations")
    _no_witness_read(monkeypatch)
    form = _at(conn, ops, 800)
    v = _draft(conn, ops, form, base=ops.live)
    assert repo.get_policy(conn, ops.key)["secondReview"] == NONE
    assert _code(conn, ops, v, REVIEWER) == "not_needed"
    assert _activate(conn, ops, form, base=v) == v + 1 and _reviews(conn, ops) == []


def test_a_policy_outside_a_protected_area_is_not_covered_even_as_it_moves_into_one(conn, make):
    # the rule protects what is live in a protected area; Operations is not one
    ops = make("Operations")
    assert _activate(conn, ops, _at(conn, ops, 800, businessArea="Finance"), base=ops.live) == 3


def test_tightening_or_an_unrelated_edit_needs_no_review_in_finance(conn, fin):
    assert _activate(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live) == 3
    assert _activate(conn, fin, _at(conn, fin, 400), base=3) == 4
    assert _activate(conn, fin, _at(conn, fin, 400, owner=f"TST Other Owner {fin.tag}"), base=4) == 5
    assert _policy(conn, fin) == (5, "live", 5, 5) and _reviews(conn, fin) == []


def test_moving_the_policy_out_of_finance_in_the_same_request_does_not_avoid_the_review(conn, fin):
    moved = _at(conn, fin, 800, businessArea="Operations")
    assert _refused(conn, fin, moved, base=fin.live) == REQUIRED
    v = _draft(conn, fin, moved, base=fin.live)
    assert _refused(conn, fin, moved, base=v) == REQUIRED
    assert repo.get_policy(conn, fin.key)["secondReview"]["required"] is True


# ------------------------------------------------------------------ who may review
def test_the_person_who_saved_the_draft_cannot_review_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live, actor="tst-saver")
    assert _code(conn, fin, v, SAVER) == "own_change"
    assert _code(conn, fin, v, person("TST-Saver")) == "own_change", "the same subject, whatever its case"
    assert _code(conn, fin, v, person("another-subject", email="tst-saver")) == "own_change", \
        "saved under a name that is this caller's email"
    assert _reviews(conn, fin) == []


def test_the_owner_cannot_review_it(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    assert _code(conn, fin, v, fin.owner_person) == "owner"
    assert _reviews(conn, fin) == []
    _review(conn, fin, v)                                   # someone linked to nobody may


def test_naming_another_owner_in_the_draft_does_not_let_the_owner_review(conn, fin):
    other = f"TST Other Owner {fin.tag}"
    conn.cursor().execute("INSERT INTO proc.bp_policy_decider_map (decider_name, emails, last_modified_by) "
                          "VALUES (%s, %s, 'test')", (other, ["tst-someone-else@example.test"]))
    v = _draft(conn, fin, _at(conn, fin, 800, owner=other), base=fin.live)
    assert _code(conn, fin, v, fin.owner_person) == "owner", "the owner of the version in force"
    assert _code(conn, fin, v, person("tst-someone-else")) == "owner", "and the owner the draft names"
    assert _reviews(conn, fin) == []


def test_an_owner_linked_to_nobody_means_no_review_can_be_recorded(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    conn.cursor().execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = %s", (fin.owner,))
    assert _code(conn, fin, v, REVIEWER) == "owner_unmapped"
    assert _reviews(conn, fin) == [], "nobody is let through because the rule cannot be evaluated"


def test_a_sign_in_without_an_email_cannot_review_when_the_owner_is_linked_by_email(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    for email in (None, "", "   "):
        nameless = Principal(subject="tst-no-email", email=email, claims={"cognito:groups": []})
        assert _code(conn, fin, v, nameless) == "identity_incomplete", repr(email)
    assert _reviews(conn, fin) == []
    # the owner linked by group only: an email could not have told us anything
    conn.cursor().execute("UPDATE proc.bp_policy_decider_map SET emails = '{}', groups = %s WHERE decider_name = %s",
                          ([f"TST-GROUP-{fin.tag}"], fin.owner))
    in_group = Principal(subject="tst-in-group", email=None, claims={"cognito:groups": [f"TST-GROUP-{fin.tag}"]})
    assert _code(conn, fin, v, in_group) == "owner"
    _review(conn, fin, v, Principal(subject="tst-no-email", email=None, claims={"cognito:groups": []}))


def test_a_review_is_refused_when_none_is_needed_or_the_version_is_not_the_latest_draft(conn, fin):
    assert _code(conn, fin, fin.live, REVIEWER) == "not_latest", "the latest version is live, not a draft"
    v = _draft(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live)
    assert _code(conn, fin, v, REVIEWER) == "not_needed", "nothing a person decided is let go"
    assert _code(conn, fin, fin.live, REVIEWER) == "not_latest"
    assert _code(conn, fin, v + 5, REVIEWER) == "not_latest"
    tighter = _draft(conn, fin, _at(conn, fin, 400), base=v)
    assert _code(conn, fin, tighter, REVIEWER) == "not_needed", "a tightening"
    with pytest.raises(L.Refused) as exc:
        SR.record(conn, "TST-nosuch", 1, principal=REVIEWER, note=None, now=NOW)
    assert exc.value.code == "not_found"
    assert _reviews(conn, fin) == []


# ------------------------------------------------------------------ fail closed
def test_activation_is_refused_when_the_suggestions_cannot_be_read(conn, fin, monkeypatch):
    form = _at(conn, fin, 400)                               # a tightening: it would need no review at all
    real = SR._SUGGESTION_SQL
    monkeypatch.setattr(SR, "_SUGGESTION_SQL", "SELECT suggestion_id FROM proc.bp_tst_no_such_table "
                                               "WHERE policy_key = %s")
    assert _refused(conn, fin, form, base=fin.live) == UNAVAILABLE
    assert conn.autocommit is True
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "no draft: the page has nothing to work out"
    monkeypatch.setattr(SR, "_SUGGESTION_SQL", real)
    assert _activate(conn, fin, form, base=fin.live) == 3, "the connection was left usable"


def test_activation_is_refused_when_the_business_area_cannot_be_read(conn, fin, monkeypatch):
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    _review(conn, fin, v)                                    # even a recorded review does not help
    monkeypatch.setattr(SR, "_SETTING_SQL", "SELECT second_reviewer FROM proc.bp_tst_no_such_table WHERE area_name = %s")
    assert _refused(conn, fin, form, base=v) == UNAVAILABLE
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}, "the page fails closed too"


def test_activation_is_refused_when_the_decisions_cannot_be_read(conn, fin, monkeypatch):
    def boom(cur, key):
        raise RuntimeError("the firing log cannot be read")
    monkeypatch.setattr(SR.learning, "witnesses", boom)
    assert _refused(conn, fin, _at(conn, fin, 400), base=fin.live) == UNAVAILABLE, "even a tightening: we cannot tell"


def test_no_review_is_recorded_when_the_decider_map_cannot_be_read(conn, fin, monkeypatch):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)

    def boom(c):
        raise RuntimeError("the map cannot be read")
    monkeypatch.setattr(SR.deciders, "load_map", boom)
    assert _code(conn, fin, v, REVIEWER) == "second_review_unavailable"
    assert _reviews(conn, fin) == [] and conn.autocommit is True


# ------------------------------------------------------------------ the policy page
def test_the_policy_page_says_a_review_is_required_and_then_who_did_it(conn, fin):
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "nothing newer than the live version"
    form = _at(conn, fin, 800)
    v = _draft(conn, fin, form, base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedAt": None}
    _review(conn, fin, v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v,
                                                              "reviewedBy": "tst-reviewer",
                                                              "reviewedAt": NOW.isoformat()}
    _activate(conn, fin, form, base=v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "it is live: nothing waits for a review"


def test_the_policy_page_asks_for_nothing_on_a_draft_that_does_not_loosen(conn, fin):
    _draft(conn, fin, _at(conn, fin, 400), base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE


# ------------------------------------------------------------------ through the whole app
def _hdr(sub):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": f"{sub}@example.test", "X-User-Groups": "[]"}


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(R, "_role_of", lambda p: ROLES.get(p.subject, "Nobody"))
    return TestClient(app)


def test_the_endpoint_needs_the_approver_role(client, conn, fin):
    v = _draft(conn, fin, _at(conn, fin, 800), base=fin.live)
    r = client.post(f"/agent-policies/{fin.key}/second-review", json={"version": v}, headers=_hdr("tst-buyer"))
    assert r.status_code == 403 and _reviews(conn, fin) == []


def test_the_whole_flow_through_the_app(client, conn, fin):
    form = _at(conn, fin, 800)
    versions, review = f"/agent-policies/{fin.key}/versions", f"/agent-policies/{fin.key}/second-review"
    r = client.post(versions, headers=_hdr("tst-saver"),
                    json={"form": form, "baseVersion": fin.live, "intent": "draft", "changeNote": "TST"})
    assert r.status_code == 200 and r.json()["version"] == 3, r.text
    activate = {"form": form, "baseVersion": 3, "intent": "activate", "changeNote": "TST"}
    r = client.post(versions, headers=_hdr("tst-saver"), json=activate)
    assert r.status_code == 422 and r.json() == {"problems": REQUIRED}, "refused, and the message arrives readable"
    page = client.get(f"/agent-policies/{fin.key}", headers=_hdr("tst-saver")).json()
    assert page["secondReview"] == {"required": True, "version": 3, "reviewedBy": None, "reviewedAt": None}
    r = client.post(review, json={"version": 3}, headers=_hdr("tst-saver"))
    assert r.status_code == 409 and r.json() == {"code": "own_change", "message": SR.REFUSALS["own_change"]}
    r = client.post(review, json={"version": 2}, headers=_hdr("tst-reviewer"))
    assert r.status_code == 409 and r.json()["code"] == "not_latest"
    r = client.post(review, json={"version": 3, "note": "TST looked"}, headers=_hdr("tst-reviewer"))
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"policyKey", "version", "reviewedBy", "reviewedAt"}
    assert (r.json()["policyKey"], r.json()["version"], r.json()["reviewedBy"]) == (fin.key, 3, "tst-reviewer")
    [(version, sid, by, _at_, note)] = _reviews(conn, fin)
    assert (version, sid, by, note) == (3, fin.sid, "tst-reviewer", "TST looked")
    page = client.get(f"/agent-policies/{fin.key}", headers=_hdr("tst-saver")).json()
    assert (page["secondReview"]["reviewedBy"], page["secondReview"]["reviewedAt"]) == ("tst-reviewer", r.json()["reviewedAt"])
    # the reviewed form exactly as GET /{key} returns it, sent back unchanged, is the reviewed change
    shown = page["versions"][-1]["form"]
    r = client.post(versions, headers=_hdr("tst-saver"),
                    json={"form": {**shown, "messageForAgent": "TST reworded"}, "baseVersion": 3, "intent": "activate",
                          "changeNote": "TST"})
    assert r.status_code == 422 and r.json() == {"problems": REQUIRED}, "one other field changed: not what was reviewed"
    r = client.post(versions, headers=_hdr("tst-saver"),
                    json={"form": shown, "baseVersion": 3, "intent": "activate", "changeNote": "TST other note"})
    assert r.status_code == 200 and r.json()["version"] == 4, r.text
    assert _policy(conn, fin) == (4, "live", 4, 4)
    assert client.post("/agent-policies/ZZZ-999999/second-review", json={"version": 1},
                       headers=_hdr("tst-reviewer")).status_code == 404
