import json, os, pytest
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB")
import psycopg2.errors
from src.services.db import get_conn

def _one(sql, *a):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(sql, a); return cur.fetchone()

def _j(v):
    return v if isinstance(v, dict) else json.loads(v)

def test_email_policy_seeded_once_with_safe_defaults():
    row = _one("SELECT policy_details, version FROM proc.bp_policy WHERE policy_name='email_policy' AND policy_status=1")
    d = _j(row[0])
    assert d["policy_identifier"] == "email_policy"
    assert d["rules"]["internal_send"] == "with_user_approval"
    assert d["rules"]["on_disconnect"] == "hold"
    assert d["rules"]["read_scope"] == "replies_and_suppliers"
    assert d["history"] == []

def _insert_expecting_check_violation(kind, status):
    with get_conn() as c:
        c.autocommit = False
        try:
            with c.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute("INSERT INTO proc.bp_internal_email(author,kind,recipients,subject,body,content_hash,status)"
                            " VALUES ('a@x',%s,'[]','s','b','h',%s)", (kind, status))
        finally:
            c.rollback()

def test_internal_email_status_check():
    _insert_expecting_check_violation("forward", "bogus")

def test_internal_email_kind_check():
    _insert_expecting_check_violation("bogus", "pending")

def test_security_reviewer_role_defined_and_mapped():
    rd = _j(_one("SELECT policy_details FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='role_definition' AND policy_status=1")[0])
    ra = _j(_one("SELECT policy_details FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='role_assignment' AND policy_status=1")[0])
    roles = rd["rules"]["roles"]
    assert roles["SecurityReviewer"]["allow"] == ["inspect"]
    assert "inspect" in roles["Admin"]["allow"]
    assert "inspect" in rd["rules"]["reversible_classes"]
    assert ra["rules"]["group_to_role"]["PROCWISE_SECURITY_REVIEWER"] == "SecurityReviewer"
