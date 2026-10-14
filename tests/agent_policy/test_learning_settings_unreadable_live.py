"""The thresholds endpoint and a body jsonb refuses, through the REAL audit write (bp_testdb).
Needs PROCWISE_TEST_LIVE_DB=1.

Nothing is faked on the audit path: `agent_actions.record_action_or_fail` writes its rows to
proc.bp_agent_actions. The request is refused (422), so the settings row is never touched. The
two audit rows it writes stay (the action log is an event log); they name a sign-in id of this
test's own.
"""
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import settings as S
from services.db import get_conn

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
URL = "/agent-policies/settings/learning"
GOOD = dict(S.DEFAULTS["learning"])


def _row(conn):
    cur = conn.cursor()
    cur.execute("SELECT config_value, last_modified_by, last_modified_date FROM proc.bp_admin_config "
                "WHERE config_key = %s", (S.CONFIG_KEY,))
    return cur.fetchone()


def _audits(conn, sub):
    cur = conn.cursor()
    cur.execute("SELECT phase, status, details FROM proc.bp_agent_actions WHERE action_type = 'agent_policy.admin' "
                "AND details ->> 'principal' = %s ORDER BY 1 DESC", (sub,))
    return [(p, s, d if isinstance(d, dict) else json.loads(d)) for p, s, d in cur.fetchall()]


@pytest.mark.parametrize("token,shown,field,message", [
    ("NaN", "nan", "wilson_lower", "Enter a number from 0.5 to 0.99."),
    ("Infinity", "inf", "min_decisions", "Enter a whole number from 10 to 1000."),
    ("1" + "0" * 400, ("1" + "0" * 400)[:100], "min_decisions", "Enter a whole number from 10 to 1000."),
])
def test_a_number_that_cannot_be_read_is_422_through_the_real_audit_write(monkeypatch, token, shown, field, message):
    from api.main import app
    assert os.getenv("DB_NAME") == "bp_testdb"
    sub = f"tst-admin-{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_role_of", lambda p: "Admin")
    body = json.dumps({**GOOD, field: "TOKEN"}).replace('"TOKEN"', token)
    with get_conn() as conn:
        before = _row(conn)
        r = TestClient(app).put(URL, content=body, headers={
            "X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": "", "X-User-Groups": "[]",
            "Content-Type": "application/json"})
        assert r.status_code == 422, r.text
        assert r.json() == {"problems": [{"field": field, "message": message}]}
        assert _row(conn) == before, "the settings row was not touched"
        (p1, s1, d1), (p2, s2, d2) = _audits(conn, sub)
        assert (p1, s1) == ("authorize", "allowed") and d1["requested"] == {**GOOD, field: shown}
        assert (p2, s2, d2["fields"]) == ("act", "refused", [field])


def _put(monkeypatch, body):
    from api.main import app
    assert os.getenv("DB_NAME") == "bp_testdb"
    sub = f"tst-admin-{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_role_of", lambda p: "Admin")
    r = TestClient(app).put(URL, content=json.dumps(body), headers={        # ascii: the escape goes on the wire
        "X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": "", "X-User-Groups": "[]",
        "Content-Type": "application/json"})
    return sub, r


@pytest.mark.parametrize("value,shown", [("\u0000", "'\\x00'"), ("\ud800", "'\\ud800'"), ("a\u0000b", "'a\\x00b'")])
def test_text_jsonb_refuses_is_422_through_the_real_audit_write(monkeypatch, value, shown):
    with get_conn() as conn:
        before = _row(conn)
        sub, r = _put(monkeypatch, {**GOOD, "min_days": value})
        assert r.status_code == 422, r.text
        assert r.json() == {"problems": [{"field": "min_days", "message": "Enter a whole number from 1 to 365."}]}
        assert _row(conn) == before, "the settings row was not touched"
        (p1, s1, d1), (p2, s2, d2) = _audits(conn, sub)
        assert (p1, s1) == ("authorize", "allowed") and d1["requested"] == {**GOOD, "min_days": shown}
        assert (p2, s2, d2["fields"]) == ("act", "refused", ["min_days"])


def test_unknown_keys_are_422_through_the_real_audit_write_twenty_names_kept(monkeypatch):
    with get_conn() as conn:
        before = _row(conn)
        sub, r = _put(monkeypatch, {**GOOD, "a\u0000": 1})
        assert r.status_code == 422, r.text
        assert [x["message"] for x in r.json()["problems"]] == ["This is not a learning threshold."]
        (_p1, s1, _d1), (p2, s2, d2) = _audits(conn, sub)
        assert s1 == "allowed" and (p2, s2, d2["fields"]) == ("act", "refused", ["'a\\x00'"])

        sub, r = _put(monkeypatch, {**GOOD, **{f"k{i:02d}": i for i in range(50)}})
        assert r.status_code == 422 and len(r.json()["problems"]) == 50
        (_p1, s1, _d1), (p2, s2, d2) = _audits(conn, sub)
        assert s1 == "allowed" and (p2, s2) == ("act", "refused")
        assert d2["fields"] == [f"k{i:02d}" for i in range(20)]
        assert _row(conn) == before, "the settings row was not touched"
