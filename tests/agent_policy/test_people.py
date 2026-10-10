"""A sign-in id resolves to a name wherever it is shown (stage 5b, Task B7). No database.

`people.remember` keeps (sign-in id -> email) for every agent-policy WRITE made through the
gateway; `people.names` resolves a set of ids with one query. Remembering never fails a person's
action and never happens on a read.
"""
import json
import logging
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.auth import Principal
from api.routers import agent_policies as R
from services.agent_policy import approval_views as V
from services.agent_policy import people as P
from services.agent_policy import second_review as SR
from services.agent_policy import settings as S
from tests.agent_policy.test_second_review import COMPILED, FINANCE, at, form

pytestmark = pytest.mark.people          # the real remember runs here (conftest makes it a no-op elsewhere)
AT = datetime(2026, 10, 10, 9, tzinfo=timezone.utc)


def person(sub="sub-1", email="Sam@Example.test"):
    return Principal(subject=sub, email=email, claims={})


class _Cur:
    def __init__(self, rows=(), fail=None, connection=None):
        self.rows, self.fail, self.ran, self.connection = list(rows), fail, [], connection

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.ran.append((sql, params))
        if self.fail and "bp_agent_policy_person" in sql:
            raise self.fail

    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, autocommit=True, **kw):
        self.autocommit, self.cur = autocommit, _Cur(**kw)
        self.cur.connection = self

    def cursor(self): return self.cur


# ------------------------------------------------------------------ remember
def test_a_person_with_an_id_and_an_email_is_remembered_in_one_statement():
    for target in (_Conn(), _Conn().cur):
        P.remember(target, person(" sub-1 ", "  Sam@Example.test "))
        cur = getattr(target, "cur", target)
        [(sql, params)] = cur.ran
        assert sql.startswith("INSERT INTO proc.bp_agent_policy_person (subject, email)")
        assert "ON CONFLICT (subject) DO UPDATE SET email = EXCLUDED.email, last_seen_at = now()" in sql
        assert "first_seen_at" not in sql.split("DO UPDATE")[1], "when they were first seen is never rewritten"
        assert params == ("sub-1", "Sam@Example.test"), "trimmed, as given: not lower-cased"


@pytest.mark.parametrize("email", [None, "", "   ", "\t\n", 0, False])
def test_a_blank_email_writes_nothing_so_it_can_never_erase_a_stored_one(email):
    conn = _Conn()
    P.remember(conn, person(email=email))
    assert conn.cur.ran == []


@pytest.mark.parametrize("who", [person(sub=""), person(sub="   "), person(sub=None), None, object()])
def test_nobody_is_remembered_without_a_sign_in_id(who):
    conn = _Conn()
    P.remember(conn, who)
    assert conn.cur.ran == []


def test_a_system_actor_is_never_remembered_as_a_person():
    conn = _Conn()
    P.remember(conn, person(sub="system:timeout", email="x@example.test"))
    assert conn.cur.ran == []


def test_the_email_is_capped_at_320_characters():
    conn = _Conn()
    P.remember(conn, person(email="a" * 400 + "@example.test"))
    assert len(conn.cur.ran[0][1][1]) == 320 == P.MAX_EMAIL


def test_a_failed_write_is_logged_once_and_never_raised(caplog):
    conn = _Conn(fail=RuntimeError("relation does not exist"))
    with caplog.at_level(logging.WARNING, logger=P.__name__):
        assert P.remember(conn, person()) is None
    said = [r for r in caplog.records if r.name == P.__name__]
    assert len(said) == 1 and "sub-1" in said[0].getMessage() and "RuntimeError" in said[0].getMessage()
    assert "Sam@Example.test" not in said[0].getMessage(), "the address is not copied into the log"

    class _NoCursor:
        autocommit = True
        def cursor(self): raise RuntimeError("connection is closed")
    assert P.remember(_NoCursor(), person()) is None


def test_inside_a_transaction_a_failed_write_is_undone_on_its_own():
    """On a connection that is in a transaction a failed statement would poison the person's own
    work: the write goes in a savepoint, and a failure rolls back to it."""
    ok = _Conn(autocommit=False)
    P.remember(ok, person())
    assert [s for s, _ in ok.cur.ran][0] == "SAVEPOINT agent_policy_person"
    assert [s for s, _ in ok.cur.ran][-1] == "RELEASE SAVEPOINT agent_policy_person"
    bad = _Conn(autocommit=False, fail=RuntimeError("boom"))
    P.remember(bad, person())
    assert [s for s, _ in bad.cur.ran][0] == "SAVEPOINT agent_policy_person"
    assert [s for s, _ in bad.cur.ran][-2:] == ["ROLLBACK TO SAVEPOINT agent_policy_person",
                                               "RELEASE SAVEPOINT agent_policy_person"]
    assert not any("SAVEPOINT" in s for s, _ in _Conn().cur.ran)


# ------------------------------------------------------------------ names
def test_names_is_one_query_for_the_whole_set():
    cur = _Cur(rows=[("sub-1", "sam@example.test"), ("sub-2", "kim@example.test")])
    got = P.names(cur, ["sub-2", "sub-1", "sub-1", "sub-9", None, "", "system:timeout", 7])
    assert got == {"sub-1": "sam@example.test", "sub-2": "kim@example.test"}
    [(sql, params)] = cur.ran
    assert sql == "SELECT subject, email FROM proc.bp_agent_policy_person WHERE subject = ANY(%s)"
    assert params == (["sub-1", "sub-2", "sub-9"],), "people only, each once: no system actor, no blank"


def test_names_asks_nothing_when_there_is_nobody_to_look_up():
    cur = _Cur()
    assert P.names(cur, []) == {} and P.names(cur, [None, "", "system:group"]) == {} and cur.ran == []


def test_names_that_cannot_be_read_are_no_names_never_an_error(caplog):
    cur = _Cur(fail=RuntimeError("relation does not exist"))
    with caplog.at_level(logging.WARNING, logger=P.__name__):
        assert P.names(cur, ["sub-1"]) == {}
    assert len([r for r in caplog.records if r.name == P.__name__]) == 1


# ------------------------------------------------------------------ the names where they are shown
FIRING = {"firing_id": 1, "policy_key": "FIN-0012", "policy_version": 3, "checkpoint": "tool.call.before",
          "action_name": "refund.issue", "agent": "a", "workflow_id": "w", "requested_by": "req", "outcome": "approve",
          "result": "approved", "matched_values": {}, "missing_inputs": [], "decision_id": 9, "decided_level": 0,
          "decided_by": "sub-1", "decided_at": AT, "reason": None, "duration_ms": 1, "reversal_of": None,
          "created_at": AT}


@pytest.mark.parametrize("decided_by,known,shown", [
    ("sub-1", {"sub-1": "sam@example.test"}, "sam@example.test"),
    ("sub-1", {}, "sub-1"),
    ("sub-1", {"sub-2": "kim@example.test"}, "sub-1"),
    ("sub-1", {"sub-1": "   "}, "sub-1"),
    ("system:timeout", {"system:timeout": "x@example.test"}, "Timeout"),
    ("system:precedent", {}, "Precedent"),
    (None, {"sub-1": "sam@example.test"}, None),
])
def test_a_firing_shows_who_decided_by_name_when_the_name_is_known(decided_by, known, shown):
    view = V._firing_view({**FIRING, "decided_by": decided_by}, set(), names=known)
    assert (view["decidedBy"], view["decidedByName"]) == (decided_by, shown)
    # with no lookup at all a person is their id, and the system keeps its words
    unnamed = V._firing_view({**FIRING, "decided_by": decided_by}, set())["decidedByName"]
    assert unnamed == (shown if str(decided_by or "").startswith("system:") else decided_by)


@pytest.mark.parametrize("stored,known,shown", [
    ("kim@stored.test", {"sub-7": "kim@lookup.test"}, "kim@stored.test"),      # the email kept with the row wins
    (None, {"sub-7": "kim@lookup.test"}, "kim@lookup.test"),                   # then the lookup
    ("  ", {"sub-7": "kim@lookup.test"}, "kim@lookup.test"),
    (None, {}, "sub-7"),                                                       # then the id
])
def test_a_reversal_prefers_the_email_stored_with_it_then_the_lookup_then_the_id(stored, known, shown):
    row = {**FIRING, "rv_by": "sub-7", "rv_email": stored, "rv_at": AT, "rv_reason": "clawed back"}
    assert V._firing_view(row, set(), names=known)["reversed"] == {
        "by": "sub-7", "byName": shown, "at": AT.isoformat(), "reason": "clawed back"}


class _FiringsCur(_Cur):
    """policy_firings' statements: the rows, the sensitive fields (two reads), then the names."""
    def __init__(self, firing_rows, people):
        super().__init__()
        self.firing_rows, self.people, self.description = firing_rows, people, None

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.ran.append((sql, params))
        if "FROM proc.bp_policy_firing f" in sql:
            cols = list(self.firing_rows[0]) if self.firing_rows else []
            self.description, self.rows = [(c,) for c in cols], [tuple(r.values()) for r in self.firing_rows]
        elif "bp_agent_policy_person" in sql:
            self.rows = [(s, e) for s, e in self.people.items() if s in params[0]]
        else:
            self.rows = []


def test_the_activity_list_resolves_every_name_with_one_query_never_one_per_row():
    rv = {"rv_by": None, "rv_email": None, "rv_at": None, "rv_reason": None}
    rows = [{**FIRING, "firing_id": 5, "decided_by": "sub-1", **rv},
            {**FIRING, "firing_id": 4, "decided_by": "sub-2", "rv_by": "sub-3", "rv_email": None, "rv_at": AT,
             "rv_reason": "r"},
            {**FIRING, "firing_id": 3, "decided_by": "sub-1", "rv_by": "sub-4", "rv_email": "four@stored.test",
             "rv_at": AT, "rv_reason": "r"},
            {**FIRING, "firing_id": 2, "decided_by": "system:timeout", **rv},
            {**FIRING, "firing_id": 1, "decided_by": None, **rv}]
    conn = _Conn()
    conn.cur = _FiringsCur(rows, {"sub-1": "sam@example.test", "sub-3": "three@example.test",
                                  "sub-4": "four@lookup.test"})
    got = V.policy_firings(conn, "FIN-0012", 50)
    assert [(f["decidedBy"], f["decidedByName"]) for f in got] == [
        ("sub-1", "sam@example.test"), ("sub-2", "sub-2"), ("sub-1", "sam@example.test"),
        ("system:timeout", "Timeout"), (None, None)]
    assert [f["reversed"] and f["reversed"]["byName"] for f in got] == [None, "three@example.test", "four@stored.test",
                                                                       None, None]
    asked = [(s, p) for s, p in conn.cur.ran if "bp_agent_policy_person" in s]
    assert len(asked) == 1, "one names query for the answer"
    assert asked[0][1] == (["sub-1", "sub-2", "sub-3"],), "sub-4 has its email with the row: not looked up"


def test_an_activity_list_with_nobody_to_name_asks_for_no_names():
    rv = {"rv_by": None, "rv_email": None, "rv_at": None, "rv_reason": None}
    conn = _Conn()
    conn.cur = _FiringsCur([{**FIRING, "decided_by": "system:timeout", **rv}, {**FIRING, "decided_by": None, **rv}], {})
    assert [f["decidedByName"] for f in V.policy_firings(conn, "FIN-0012", 50)] == ["Timeout", None]
    assert not any("bp_agent_policy_person" in s for s, _ in conn.cur.ran)


class _Scripted:
    def __init__(self, answers):
        self.answers, self.asked, self._rows, self.connection = list(answers), [], [], None

    def execute(self, sql, params=None):
        self.asked.append(" ".join(sql.split()))
        self._rows = self.answers.pop(0)

    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None


@pytest.mark.parametrize("stored,people,shown,lookups", [
    ("r@stored.test", [("tst-reviewer", "r@lookup.test")], "r@stored.test", 0),   # the row's own email: no lookup
    (None, [("tst-reviewer", "r@lookup.test")], "r@lookup.test", 1),
    (None, [], "tst-reviewer", 1),
])
def test_the_second_reviewer_is_named_from_the_row_then_the_lookup_then_the_id(monkeypatch, stored, people, shown,
                                                                               lookups):
    monkeypatch.setattr(SR, "load_settings", lambda conn: S.merge(None))
    answers = [[(7,)], [(form(), COMPILED)], *FINANCE, *FINANCE, [(form(at(800)), "tst-reviewer", AT, stored)]]
    cur = _Scripted(answers + ([people] if lookups else []))
    versions = [{"version": 2, "savedAs": "live", "form": form()},
                {"version": 3, "savedAs": "draft", "form": form(at(800))}]
    got = SR.status(cur, "FIN-0001", {"liveVersion": 2, "latestVersion": 3}, versions)
    assert (got["reviewedBy"], got["reviewedByName"]) == ("tst-reviewer", shown)
    assert len([q for q in cur.asked if "bp_agent_policy_person" in q]) == lookups


# ------------------------------------------------------------------ through the whole app: writes remember, reads do not
HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": " U1@Example.test ", "X-User-Groups": json.dumps([])}
ROLES = {"u1": "Admin", "u-viewer": "Viewer"}


@pytest.fixture
def app_state(monkeypatch):
    from api.main import app
    state = {"remembered": [], "conns": 0, "audits": []}
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_role_of", lambda p: ROLES.get(p.subject, "Nobody"))
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: state["audits"].append(kw))

    class _Ctx:
        def __enter__(self):
            state["conns"] += 1
            return _Conn()
        def __exit__(self, *a): return False

    monkeypatch.setattr(R, "_conn", lambda: _Ctx())
    monkeypatch.setattr(R.people, "remember", lambda conn, p: state["remembered"].append((p.subject, p.email)))
    monkeypatch.setattr(R.approval_views, "mark_read", lambda conn, nid, p, is_admin=False: {"id": nid, "read": True})
    monkeypatch.setattr(R.approval_views, "my_notifications", lambda conn, p, limit, is_admin=False: [])
    monkeypatch.setattr(R.approval_views, "list_deciders", lambda conn: [])
    monkeypatch.setattr(R.approval_views, "list_cases", lambda conn, p, is_admin, status="open": [])
    monkeypatch.setattr(R.approval_views, "policy_firings", lambda conn, key, limit: [])
    monkeypatch.setattr(R.conflict_views, "list_conflicts", lambda conn, p, status, is_admin: [])
    monkeypatch.setattr(R.repo, "list_policies", lambda conn: [])
    monkeypatch.setattr(R.documents, "list_documents", lambda conn: [])
    monkeypatch.setattr(R.run_store, "list_recent", lambda conn: [])
    monkeypatch.setattr(R.learning, "list_cards", lambda conn, status="open", caller=None: [])
    monkeypatch.setattr(R.learning, "caller_of", lambda conn, p, editor, approver, mapping=None: R.learning.Caller())
    monkeypatch.setattr(R.policy_settings, "load_settings", lambda conn=None: S.merge(None))
    return TestClient(app), state


WRITE = "/agent-policies/notifications/7/read"
GETS = ["/agent-policies", "/agent-policies/notifications", "/agent-policies/deciders", "/agent-policies/approvals",
        "/agent-policies/conflicts", "/agent-policies/suggestions", "/agent-policies/documents",
        "/agent-policies/extraction-runs", "/agent-policies/FIN-0012/firings", "/agent-policies/settings/learning"]


def test_a_write_remembers_the_caller_once(app_state):
    client, state = app_state
    assert client.post(WRITE, headers=HDR).status_code == 200
    assert state["remembered"] == [("u1", " U1@Example.test ")], "once per request; remember trims it"


@pytest.mark.parametrize("url", GETS)
def test_a_get_does_not_write(app_state, url):
    client, state = app_state
    r = client.get(url, headers=HDR)
    assert r.status_code == 200, r.text
    assert state["remembered"] == [], "a read never remembers anybody"
    assert state["conns"] <= 1, "and opens no connection of its own for it"


def test_no_get_route_can_reach_remember():
    """Every GET handler asks `_require` for agent_policy.read, and `_require` remembers nobody for
    that action: pinned on the source, so a new GET cannot quietly start writing."""
    import inspect
    for route in R.router.routes:
        if "GET" in route.methods:
            source = inspect.getsource(route.endpoint)
            assert '"agent_policy.read"' in source and source.count("_require(") == 1, route.path
    required = inspect.getsource(R._require)
    assert required.count("_remember(") == 1
    assert 'if action != "agent_policy.read":\n        _remember(principal)' in required


def test_a_refused_role_is_not_remembered(app_state):
    client, state = app_state
    r = client.put("/agent-policies/settings/learning", json={}, headers={**HDR, "X-User-Sub": "u-viewer"})
    assert r.status_code == 403 and state["remembered"] == []


def test_a_sign_in_with_no_email_opens_no_connection_to_remember_nothing(app_state):
    client, state = app_state
    assert client.post(WRITE, headers={**HDR, "X-User-Email": "  "}).status_code == 200
    assert state["remembered"] == [] and state["conns"] == 1, "only the handler's own connection"


@pytest.mark.parametrize("how", ["remember raises", "the connection cannot be opened"])
def test_remembering_that_fails_does_not_fail_the_request(app_state, monkeypatch, caplog, how):
    client, state = app_state
    if how == "remember raises":
        def boom(conn, p):
            raise RuntimeError("relation proc.bp_agent_policy_person does not exist")
        monkeypatch.setattr(R.people, "remember", boom)
    else:
        real, calls = R._conn, []

        def flaky():
            calls.append(1)
            if len(calls) == 1:                      # the first connection of the request is remember's
                raise RuntimeError("could not connect")
            return real()
        monkeypatch.setattr(R, "_conn", flaky)
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        r = client.post(WRITE, headers=HDR)
    assert r.status_code == 200 and r.json() == {"id": 7, "read": True}, r.text
    assert any("could not be remembered" in rec.getMessage() for rec in caplog.records)


def test_the_real_remember_failing_on_a_real_cursor_does_not_fail_the_request(app_state, monkeypatch):
    client, state = app_state
    monkeypatch.setattr(R.people, "remember", P.remember)        # the real one, on a cursor that fails

    class _Ctx:
        def __enter__(self): return _Conn(fail=RuntimeError("boom"))
        def __exit__(self, *a): return False
    monkeypatch.setattr(R, "_conn", lambda: _Ctx())
    assert client.post(WRITE, headers=HDR).status_code == 200
