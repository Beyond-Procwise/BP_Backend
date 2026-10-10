"""The second reviewer on bp_testdb, through the real save path. Needs PROCWISE_TEST_LIVE_DB=1.

Each test has REAL live policies ("TST …", on a tool named tst_<tag>, RETIRED at teardown): one in
Finance (second reviewer on, by seed) and, where it says so, one in Operations (off). Each gets
the standard evidence and its one suggestion from the real scan (worked in
test_learning_scan_live's docstring): 33 approvals at 510..830 under "args.amount gt 500" ->
the suggestion is 500 -> 800. The suggestion matters only in that it EXISTS: once learning has
spoken about a protected policy, any change to what it enforces needs a second person, whether
it loosens (800), tightens (400, 499) or only rewrites the condition.

readiness and the final check are bypassed (tst_ tools are unknown to the registry); every
restructured condition is therefore also put through the real conditions.to_engine (`_cond`).

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
from tests.agent_policy.test_learning_scan_live import BASE, NOW, fire

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REQUIRED = [{"field": "secondReview", "code": "second_review_required", "message": SR.MESSAGE}]
UNAVAILABLE = [{"field": "secondReview", "code": "second_review_unavailable",
                "message": "Can't check whether a second review is needed right now. Try again shortly."}]
NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}
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
        fire(conn, w.key, BASE, version=w.live, extra=extra)
        [target] = [t for t in L.targets(conn.cursor()) if t.key == w.key]
        counts = L.scan(conn, NOW, only=[target])
        assert (counts["raised"], counts["errors"]) == (2 if countries else 1, 0), counts
        w.sid = min(c["id"] for c in L.for_policy(conn.cursor(), w.key))
        return w

    yield _make
    failed = _clean_up(conn, made)
    assert not failed, f"teardown left test data behind: {failed}"


def _retire(conn, key):
    got = repo.get_policy(conn, key)
    if got["status"] != "retired":
        repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


def _clean_up(conn, made):
    """Remove what each made policy left, one step at a time: a step that fails is noted and the
    rest (of that policy and of every other) still run. Returns what failed, as text."""
    def delete(sql, value):
        conn.cursor().execute(sql, (value,))

    failed = []
    for w in made:
        steps = [("decider map", lambda: delete("DELETE FROM proc.bp_policy_decider_map WHERE decider_name LIKE %s",
                                                f"TST % {w.tag}"))]
        if w.key:
            steps += [
                ("second reviews", lambda: delete("DELETE FROM proc.bp_agent_policy_second_review "
                                                  "WHERE policy_key = %s", w.key)),
                ("suggestions", lambda: delete("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s",
                                               w.key)),
                ("retire", lambda: _retire(conn, w.key)),
            ]
        for what, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - one failure must not leave the others' data behind
                failed.append(f"{w.key or w.tag} {what}: {exc!r}")
    return failed


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


def _cycle(conn, w, form, name=""):
    """One Approver acting alone is refused (straight from the live version, and from a saved
    draft); the page says a review is required; after another Approver reviews that exact draft
    it goes Active. `w.live` moves to the new live version."""
    assert _refused(conn, w, form, base=w.live) == REQUIRED, name
    v = _draft(conn, w, form, base=w.live)
    assert _refused(conn, w, form, base=v) == REQUIRED, name
    assert repo.get_policy(conn, w.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                            "reviewedByName": None, "reviewedAt": None}, name
    _review(conn, w, v)
    assert _activate(conn, w, form, base=v) == v + 1, name
    w.live = v + 1
    assert _policy(conn, w)[1:] == ("live", w.live, w.live), name


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
        L.act(conn, fin.sid, {"dismissed": "dismiss", "not_yet": "not_yet"}[state], actor="tst-buyer", now=NOW,
              caller=L.Caller(editor=True, approver=True))
    form = _at(conn, fin, limit)
    v = _draft(conn, fin, form, base=fin.live)
    assert _refused(conn, fin, form, base=v) == REQUIRED
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedByName": None, "reviewedAt": None}
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v) == v + 1


# ------------------------------------------------------------------ the routes of three review rounds
def _wrap_any(condition):
    return {"any": [condition]}


EXISTS = {"field": "args.amount", "op": "exists"}


def test_route_1_every_restructured_condition_needs_the_review(conn, fin):
    for name, build in {
        "gte 800": lambda: _cond(conn, fin, amount("gte", 800)),
        "gt 500 AND gte 800": lambda: _cond(conn, fin, amount("gt", 500), amount("gte", 800)),
        "gt 800 listed twice": lambda: _cond(conn, fin, amount("gt", 800), amount("gt", 800)),
        "any[gt 800] inside the all": lambda: _cond(conn, fin, {"any": [amount("gt", 800)]}),
        "not lte 800": lambda: _cond(conn, fin, {"not": amount("lte", 800)}),
        "a top-level any wrapper": lambda: _cond(conn, fin, amount("gt", 800), top=_wrap_any),
    }.items():
        _cycle(conn, fin, build(), name)


def test_route_1_every_restructured_list_needs_the_review(conn, make):
    w = make("Finance", countries=True)
    de = {"field": "customer.country", "op": "in", "value": ["DE"]}
    wider = {"field": "customer.country", "op": "in", "value": ["FR", "DE", "IT"]}
    for name, rest in {
        "FR dropped, the leaf listed twice": (amount("gt", 500), de, de),
        "FR dropped, the leaf wrapped in any": (amount("gt", 500), {"any": [de]}),
        "FR dropped": (amount("gt", 500), de),
        "a country added (nothing let go: no judgement of direction)": (amount("gt", 500), wider),
    }.items():
        _cycle(conn, w, _cond(conn, w, *rest), name)


def test_route_2_a_tightening_needs_the_review_itself_so_the_two_step_is_closed_at_step_one(conn, fin):
    assert _refused(conn, fin, _at(conn, fin, 499), base=2) == REQUIRED, "step one, alone: refused"
    assert _refused(conn, fin, _cond(conn, fin, amount("gte", 500.01)), base=2) == REQUIRED
    assert _refused(conn, fin, _at(conn, fin, 400), base=2) == REQUIRED
    _cycle(conn, fin, _at(conn, fin, 499), "the tightening, reviewed")
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED, "and step two needs its own"
    _cycle(conn, fin, _at(conn, fin, 800), "the loosening, reviewed")


def test_route_2_a_superseded_suggestion_still_counts(conn, fin):
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET status = 'superseded' WHERE suggestion_id = %s",
                          (fin.sid,))
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED


def test_route_3_moving_the_policy_out_of_the_protected_area_needs_the_review_itself(conn, fin):
    moved = _at(conn, fin, businessArea="Operations")           # nothing else changed
    assert _refused(conn, fin, _at(conn, fin, businessArea="TST no such area"), base=fin.live) == REQUIRED, \
        "an unknown area resolves, as a save resolves it, to the unassigned one"
    assert _refused(conn, fin, _at(conn, fin, 800, businessArea="Operations"), base=fin.live) == REQUIRED
    _cycle(conn, fin, moved)


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
        "800 written 800.0": _at(conn, fin, 800.0),
    }
    for name, form in changed.items():
        assert _refused(conn, fin, form, base=v) == REQUIRED, name
    assert _activate(conn, fin, _at(conn, fin, 800), base=v) == v + 1


def test_route_5_changes_that_do_not_touch_the_condition(conn, fin):
    _cycle(conn, fin, _cond(conn, fin, onMissingData="fail_open"), "fail-open")
    _cycle(conn, fin, _tools(conn, fin, _at(conn, fin), [fin.tool, "tst_other"]), "a tool added")
    _cycle(conn, fin, _tools(conn, fin, _at(conn, fin), ["tst_other"]), "one of several removed")
    _cycle(conn, fin, _tools(conn, fin, _at(conn, fin), []), "no action listed")
    swapped = _at(conn, fin)
    swapped["hidden"]["condition"]["all"][0] = {"field": "tool.name", "op": "in", "value": ["tst_other"]}
    _cycle(conn, fin, swapped, "the tool replaced in the tool.name leaf")
    _cycle(conn, fin, _at(conn, fin, outcome="notify", notify=["TST Someone"]), "approve -> notify")


def test_route_5_the_company_default_switched_to_fail_open_shows_as_a_change(conn, fin, monkeypatch):
    """The live version was compiled fail-closed from a form that names no rule. With the company
    default now fail-open, saving even a reworded form would compile it fail-open."""
    cur = conn.cursor()
    cur.execute("SELECT form_state->'hidden'->>'onMissingData', compiled->'trigger'->>'onMissingData' "
                "FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s", (fin.key, fin.live))
    assert cur.fetchone() == (None, "fail_closed")
    reworded = _at(conn, fin, messageForAgent="TST reworded")
    assert SR.needed(cur, fin.key, fin.live, reworded) is None, "before the switch: a wording edit"
    real = SR.load_settings
    monkeypatch.setattr(SR, "load_settings", lambda c: {**real(c), "on_missing_data": {"approve": "fail_open"}})
    assert SR.needed(cur, fin.key, fin.live, _cond(conn, fin, onMissingData="fail_closed")) is None, \
        "the rule in force, said out loud"
    _cycle(conn, fin, reworded)


def test_route_6_nobody_who_saved_a_version_since_the_live_one_may_review(conn, fin):
    form = _at(conn, fin, 800)
    v3 = _draft(conn, fin, form, base=fin.live, actor="tst-saver")
    v4 = _draft(conn, fin, form, base=v3, actor="tst-b")
    assert _code(conn, fin, v4, SAVER) == "own_change", "A saved the change; B only re-saved it"
    assert _code(conn, fin, v4, person("tst-b")) == "own_change"
    assert _reviews(conn, fin) == []
    _review(conn, fin, v4)
    assert _activate(conn, fin, form, base=v4) == v4 + 1


@pytest.mark.parametrize("limit", [800, 400])
def test_route_7_a_retired_policy_cannot_come_back_changed_without_the_review(conn, fin, limit):
    retired = repo.retire(conn, fin.key, base_version=2, actor="tst-saver", change_note="TST retire")["version"]
    assert _policy(conn, fin) == (3, "retired", None, 3)
    form = _at(conn, fin, limit)
    assert _refused(conn, fin, form, base=retired) == REQUIRED, "measured against the last live version"
    v = _draft(conn, fin, form, base=retired, actor="tst-b")
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedByName": None, "reviewedAt": None}
    assert _code(conn, fin, v, SAVER) == "own_change", "the person who retired it saved a version since it was live"
    _review(conn, fin, v)
    assert _activate(conn, fin, form, base=v) == v + 1
    assert _policy(conn, fin) == (5, "live", 5, 5), "a retired policy CAN be made live again (stage 1 behaviour)"


def test_a_retired_policy_comes_back_unchanged_without_a_review(conn, fin):
    retired = repo.retire(conn, fin.key, base_version=2, actor="tst-saver", change_note="TST retire")["version"]
    assert _activate(conn, fin, _at(conn, fin), base=retired) == 4
    assert _policy(conn, fin)[1:] == ("live", 4, 4) and _reviews(conn, fin) == []


def test_route_8_every_missing_data_guard_needs_the_review(conn, fin):
    tool_is = {"field": "tool.name", "op": "eq", "value": fin.tool}
    b, c = {"field": "args.b", "op": "exists"}, {"field": "args.c", "op": "exists"}
    for name, rest in {
        "round 2: an exists guard": (EXISTS, amount("gt", 500)),
        "round 2: not(not exists)": ({"not": {"not": EXISTS}}, amount("gt", 500)),
        "round 3: a value-dependent guard": (amount("gt", 500), {"any": [EXISTS, tool_is]}),
        "round 3: a guard on some of the fields": (amount("gt", 500), {"any": [EXISTS, b, {"not": c}]}),
    }.items():
        _cycle(conn, fin, _cond(conn, fin, *rest), name)


def test_route_9_one_is_not_true_and_800_is_not_800_point_0(conn, fin):
    flag = {"field": "args.flag", "op": "eq", "value": 1}
    _cycle(conn, fin, _cond(conn, fin, amount("gt", 800), flag))
    _cycle(conn, fin, _cond(conn, fin, amount("gt", 800), {**flag, "value": True}), "eq 1 -> eq true")
    _cycle(conn, fin, _cond(conn, fin, amount("gt", 800.0), {**flag, "value": True}), "gt 800 -> gt 800.0")
    # and what was stored comes back as it was written: resubmitted, it is no change
    assert _activate(conn, fin, _at(conn, fin), base=fin.live) == fin.live + 1


def test_route_10_who_decides_how_fast_and_what_is_hidden(conn, fin):
    _cycle(conn, fin, _at(conn, fin, deciders=["TST Someone Else"]), "deciders")
    _cycle(conn, fin, _at(conn, fin, responseTime="PT1H"), "response time")
    flagged = _at(conn, fin)
    flagged["hidden"]["inputs"][0]["sensitive"] = not flagged["hidden"]["inputs"][0]["sensitive"]
    _cycle(conn, fin, flagged, "an input's sensitive flag")
    _cycle(conn, fin, _at(conn, fin, limit={"on": True, "text": "TST UK only"}), "limit")


def test_route_11_renaming_the_source_document_needs_the_review(conn, fin):
    """Enforcement reads `source.document`: it decides whether this policy and another on the
    same tool are tiered (each keeps its approval levels) or conflicting (last level only)."""
    live = _at(conn, fin)
    renamed = _at(conn, fin, source={**live["source"], "document": f"TST another document {fin.tag}"})
    assert {k for k in live if live[k] != renamed[k]} == {"source"}, "nothing else is changed"
    assert live["source"]["document"] != renamed["source"]["document"]
    _cycle(conn, fin, renamed, "the source document renamed")


def test_the_teardown_carries_on_past_a_step_that_fails(conn):
    """One policy that cannot be cleaned up must not leave the others' rows behind."""
    done = []

    class _Cur:
        def execute(self, sql, params=None):
            if params == ("TST-broken",) or params == ("TST % broken",):
                raise RuntimeError("cannot be removed")
            done.append((sql.split()[2], params[0]))

    fake = SimpleNamespace(cursor=_Cur)
    made = [SimpleNamespace(tag="broken", key="TST-broken"), SimpleNamespace(tag="never", key=None),
            SimpleNamespace(tag="fine", key="TST-fine")]
    retired = []
    real = globals()["_retire"]
    globals()["_retire"] = lambda c, key: retired.append(key) if key == "TST-fine" else 1 / 0
    try:
        failed = _clean_up(fake, made)
    finally:
        globals()["_retire"] = real
    assert [f.split(":")[0] for f in failed] == ["TST-broken decider map", "TST-broken second reviews",
                                                 "TST-broken suggestions", "TST-broken retire"]
    assert done == [("proc.bp_policy_decider_map", "TST % never"), ("proc.bp_policy_decider_map", "TST % fine"),
                    ("proc.bp_agent_policy_second_review", "TST-fine"),
                    ("proc.bp_agent_policy_suggestion", "TST-fine")]
    assert retired == ["TST-fine"]


def test_a_stored_suggestion_that_cannot_be_read_changes_nothing(conn, fin):
    # the suggestion's content is not consulted: only that learning has spoken
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET proposal = '{\"kind\": \"threshold\"}'::jsonb "
                          "WHERE suggestion_id = %s", (fin.sid,))
    assert _refused(conn, fin, _at(conn, fin, 800), base=fin.live) == REQUIRED
    assert _activate(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live) == 3


# ------------------------------------------------------------------ what needs no review
def test_a_wording_only_edit_needs_no_review_in_finance(conn, fin):
    live = _at(conn, fin)
    edits = {
        "name": f"TST renamed {fin.tag}",
        "situation": "TST the agent is about to refund more than 500.",
        "messageForAgent": "TST reworded",
        "messageForPerson": "TST reworded for the person",
        "owner": f"TST Other Owner {fin.tag}",
        "reviewBy": "2027-06-30",
        "examples": live["examples"][:1],
    }
    assert all(live[k] != v for k, v in edits.items())
    for key, value in edits.items():
        assert _activate(conn, fin, _at(conn, fin, **{key: value}), base=fin.live) == fin.live + 1, key
        fin.live += 1
    # all of them now differ from the first live form; another change note; then resubmitted unchanged
    assert _activate(conn, fin, _at(conn, fin, changeNote="TST another note"), base=fin.live) == fin.live + 1
    fin.live += 1
    assert _activate(conn, fin, repo.get_policy(conn, fin.key)["versions"][-1]["form"], base=fin.live) == fin.live + 1
    assert _reviews(conn, fin) == []


def test_a_policy_with_no_suggestion_needs_no_review_whatever_changes(conn, fin):
    conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (fin.key,))
    assert _activate(conn, fin, _at(conn, fin, 800), base=2) == 3
    assert _activate(conn, fin, _at(conn, fin, 400, outcome="notify", notify=["TST Someone"]), base=3) == 4
    assert _activate(conn, fin, _at(conn, fin, 400, businessArea="Operations"), base=4) == 5
    assert _reviews(conn, fin) == []


def test_operations_needs_no_second_review_whatever_changes(conn, make):
    ops = make("Operations")
    form = _at(conn, ops, 800)
    v = _draft(conn, ops, form, base=ops.live)
    assert repo.get_policy(conn, ops.key)["secondReview"] == NONE
    assert _code(conn, ops, v, REVIEWER) == "not_needed"
    assert _activate(conn, ops, form, base=v) == v + 1
    ops.live = v + 1
    assert _activate(conn, ops, _cond(conn, ops, amount("gt", 400), onMissingData="fail_open"), base=ops.live) == v + 2
    assert _reviews(conn, ops) == []


def test_a_policy_outside_a_protected_area_is_not_covered_even_as_it_moves_into_one(conn, make):
    # the rule protects what is live in a protected area; Operations is not one
    ops = make("Operations")
    assert _activate(conn, ops, _at(conn, ops, 800, businessArea="Finance"), base=ops.live) == 3


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
    assert _code(conn, fin, v, REVIEWER) == "not_needed", "nothing the policy enforces is changed"
    assert _code(conn, fin, fin.live, REVIEWER) == "not_latest"
    assert _code(conn, fin, v + 5, REVIEWER) == "not_latest"
    with pytest.raises(L.Refused) as exc:
        SR.record(conn, "TST-nosuch", 1, principal=REVIEWER, note=None, now=NOW)
    assert exc.value.code == "not_found"
    assert _reviews(conn, fin) == []


# ------------------------------------------------------------------ fail closed
def test_activation_is_refused_when_the_suggestions_cannot_be_read(conn, fin, monkeypatch):
    form = _at(conn, fin, messageForAgent="TST reworded")    # a wording edit: it would need no review at all
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
                                                              "reviewedByName": None, "reviewedAt": None}, "the page fails closed too"


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
                                                              "reviewedByName": None, "reviewedAt": None}
    _review(conn, fin, v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v,
                                                              "reviewedBy": "tst-reviewer",
                                                              "reviewedByName": "tst-reviewer@example.test",
                                                              "reviewedAt": NOW.isoformat()}
    _activate(conn, fin, form, base=v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE, "it is live: nothing waits for a review"


def test_the_policy_page_asks_for_nothing_on_a_draft_that_changes_only_wording(conn, fin):
    v = _draft(conn, fin, _at(conn, fin, messageForAgent="TST reworded"), base=fin.live)
    assert repo.get_policy(conn, fin.key)["secondReview"] == NONE
    v = _draft(conn, fin, _at(conn, fin, 400), base=v)
    assert repo.get_policy(conn, fin.key)["secondReview"] == {"required": True, "version": v, "reviewedBy": None,
                                                              "reviewedByName": None, "reviewedAt": None}, "a tightening is a change"


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
    assert page["secondReview"] == {"required": True, "version": 3, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}
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
