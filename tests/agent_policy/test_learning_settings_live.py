"""The learning thresholds row, for real, on bp_testdb. Needs PROCWISE_TEST_LIVE_DB=1.

Other sessions' tests read proc.bp_admin_config['agent_policy_settings'], so NOTHING here writes
that row: the settings module's row key is pointed at a key only this test uses
('agent_policy_settings_tst_<hex>'), and the real SQL (create if absent, lock, merge, write, the
audit in the same transaction) and `load_settings` run against that row. The test's row is deleted
at teardown, and the real row is checked to be exactly what it was.
"""
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import settings as S
from tests.agent_policy.test_learning_settings import DEFAULTS, GOOD, URL, _hdr

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REAL_KEY = "agent_policy_settings"
_ROW_SQL = "SELECT config_value::text, last_modified_by, last_modified_date FROM proc.bp_admin_config WHERE config_key = %s"


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "these tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def _row(conn, key):
    cur = conn.cursor()
    cur.execute(_ROW_SQL, (key,))
    return cur.fetchone()


def _value(conn, key):
    row = _row(conn, key)
    return None if row is None else json.loads(row[0])


@pytest.fixture
def row(conn, monkeypatch):
    """A settings row key nobody else reads. The real row must come out untouched."""
    assert S.CONFIG_KEY == REAL_KEY
    key = f"{REAL_KEY}_tst_{uuid.uuid4().hex[:10]}"
    real_before = _row(conn, REAL_KEY)
    monkeypatch.setattr(S, "CONFIG_KEY", key)
    try:
        yield key
    finally:
        conn.cursor().execute("DELETE FROM proc.bp_admin_config WHERE config_key = %s AND config_key LIKE %s",
                              (key, REAL_KEY + "_tst_%"))
        assert _row(conn, REAL_KEY) == real_before, "the shared settings row was changed"


@pytest.fixture
def api(monkeypatch):
    from api.main import app
    audits = []
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R, "_role_of", lambda p: "Admin" if p.subject.startswith("tst-admin") else "Approver")
    client = TestClient(app, raise_server_exceptions=False)
    client.audits = audits
    return client


def _seed(conn, key, value):
    conn.cursor().execute("INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
                          "VALUES (%s, %s::jsonb, 'tst-seed')", (key, json.dumps(value)))


def test_the_row_is_created_when_it_does_not_exist_and_the_next_read_has_the_new_values(api, conn, row):
    assert _row(conn, row) is None
    assert S.load_settings(conn)["learning"] == DEFAULTS, "no row: the ruled defaults"
    assert api.get(URL, headers=_hdr("tst-admin")).json()["learning"] == DEFAULTS
    r = api.put(URL, json=GOOD, headers=_hdr("tst-admin"))
    assert r.status_code == 200, r.text
    assert r.json() == {"learning": GOOD, "defaults": DEFAULTS, "scan": {"enabled": S.learning_scan_enabled()}}
    assert _value(conn, row) == {"learning": GOOD} and _row(conn, row)[1] == "tst-admin"
    loaded = S.load_settings(conn)
    assert loaded["learning"] == GOOD, "the next settings read has them"
    assert loaded["response_time"] == "PT4H" and loaded["conflict_cases_per_run"] == 25, "the rest: still the defaults"
    assert S.load_settings()["learning"] == GOOD, "and on a connection of its own"
    assert api.get(URL, headers=_hdr("tst-admin")).json()["learning"] == GOOD
    done = api.audits[-1]
    assert (done["phase"], done["status"]) == ("act", "done")
    assert (done["details"]["before"], done["details"]["after"]) == (DEFAULTS, GOOD)


def test_every_other_key_in_the_row_survives(api, conn, row):
    others = {"response_time": "PT8H", "response_time_basis": "business",
              "on_missing_data": {"approve": "fail_closed", "block": "fail_open", "notify": "fail_closed"},
              "live_conflict_repeat": 5, "conflict_cases_per_run": 7, "tst_unknown": {"nested": [1, "two", None]}}
    earlier = {**DEFAULTS, "min_decisions": 55, "tst_extra": "kept"}
    _seed(conn, row, {**others, "learning": earlier})
    r = api.put(URL, json=GOOD, headers=_hdr("tst-admin-2"))
    assert r.status_code == 200, r.text
    stored = _value(conn, row)
    assert {k: v for k, v in stored.items() if k != "learning"} == others, "nothing but learning moved"
    assert stored["learning"] == {**GOOD, "tst_extra": "kept"}, "the seven replaced; anything else under it kept"
    assert _row(conn, row)[1] == "tst-admin-2"
    done = api.audits[-1]["details"]
    assert done["before"] == {k: earlier[k] for k in DEFAULTS} and done["after"] == GOOD
    loaded = S.load_settings(conn)
    assert (loaded["learning"]["min_decisions"], loaded["response_time"], loaded["conflict_cases_per_run"]) == \
        (40, "PT8H", 7)


def test_a_row_with_no_learning_key_gains_one_and_the_audit_says_the_defaults_were_in_force(api, conn, row):
    _seed(conn, row, {"response_time": "PT2H"})
    assert api.put(URL, json=GOOD, headers=_hdr("tst-admin")).status_code == 200
    assert _value(conn, row) == {"response_time": "PT2H", "learning": GOOD}
    assert api.audits[-1]["details"]["before"] == DEFAULTS


def test_a_change_whose_audit_record_cannot_be_written_is_not_made(api, conn, row, monkeypatch):
    stored = {"response_time": "PT8H", "learning": dict(DEFAULTS)}
    _seed(conn, row, stored)
    before = _row(conn, row)

    def audit(**kw):
        if kw["phase"] == "act" and kw["status"] == "done":
            raise RuntimeError("TST audit down")

    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", audit)
    r = api.put(URL, json=GOOD, headers=_hdr("tst-admin"))
    assert r.status_code == 500 and "TST" not in r.text
    assert _row(conn, row) == before, "rolled back: value, who and when all as they were"
    assert conn.autocommit is True


def test_a_created_row_is_rolled_back_with_a_failed_audit(api, conn, row, monkeypatch):
    def audit(**kw):
        if kw["phase"] == "act" and kw["status"] == "done":
            raise RuntimeError("TST audit down")

    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", audit)
    assert api.put(URL, json=GOOD, headers=_hdr("tst-admin")).status_code == 500
    assert _row(conn, row) is None, "the row the save created is gone with it"


def test_a_stored_value_that_is_not_an_object_is_left_alone(api, conn, row):
    _seed(conn, row, ["not", "settings"])
    before = _row(conn, row)
    r = api.put(URL, json=GOOD, headers=_hdr("tst-admin"))
    assert r.status_code == 500, "nothing to merge into: an administrator must look at the row"
    assert _row(conn, row) == before


def test_a_refused_body_and_a_refused_role_write_nothing(api, conn, row):
    assert api.put(URL, json={**GOOD, "min_days": 0}, headers=_hdr("tst-admin")).status_code == 422
    assert api.put(URL, json=GOOD, headers=_hdr("tst-approver")).status_code == 403
    assert api.get(URL, headers=_hdr("tst-approver")).status_code == 403
    assert _row(conn, row) is None


def test_the_row_is_locked_while_it_is_rewritten(conn, row):
    """Two administrators at once: the second waits for the first, so neither's write is lost to a
    read made before the other's commit. Shown by holding the row and timing the save out."""
    import psycopg2

    from services.agent_policy import approvals
    from services.db import get_conn
    _seed(conn, row, {"response_time": "PT8H"})
    with get_conn() as other:
        with approvals._tx(other):
            other.cursor().execute("SELECT 1 FROM proc.bp_admin_config WHERE config_key = %s FOR UPDATE", (row,))
            conn.cursor().execute("SET lock_timeout = '500ms'")
            try:
                with pytest.raises(psycopg2.errors.LockNotAvailable):
                    S.save_learning(conn, GOOD, actor="tst-admin", record=lambda before, after: None)
            finally:
                conn.cursor().execute("RESET lock_timeout")
    assert _value(conn, row) == {"response_time": "PT8H"}
    S.save_learning(conn, GOOD, actor="tst-admin", record=lambda before, after: None)
    assert _value(conn, row) == {"response_time": "PT8H", "learning": GOOD}
