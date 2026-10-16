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


def _role_update_sql(name="2026-10-16-email-settings.sql"):
    """The role_definition UPDATE in the migration (or its rollback), as written in the file."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "migrations", name)
    text = open(path).read()
    stmts = [s for s in text.split(";") if "UPDATE proc.bp_policy" in s
             and "{rules,roles,SecurityReviewer}'" in s]
    assert len(stmts) == 1
    return stmts[0][stmts[0].index("UPDATE"):]


@pytest.mark.parametrize("details", [
    {"policy_identifier": "role_definition", "rules": {"roles": {"Admin": {"rank": 4}}}},
    {"policy_identifier": "role_definition", "rules": {"roles": {}}},
])
def test_role_update_never_nulls_a_row_missing_admin_allow(details):
    """Migration guard: a role_definition without Admin.allow must not be wiped to NULL."""
    with get_conn() as c:
        c.autocommit = False
        try:
            with c.cursor() as cur:
                cur.execute("CREATE TEMP TABLE bp_policy_probe (policy_details jsonb, version int, "
                            "last_modified_by text, last_modified_date timestamptz, "
                            "policy_status int) ON COMMIT DROP")
                cur.execute("INSERT INTO bp_policy_probe VALUES (%s::jsonb, 1, 'x', now(), 1)",
                            (json.dumps(details),))
                cur.execute(_role_update_sql().replace("proc.bp_policy", "bp_policy_probe"))
                cur.execute("SELECT policy_details FROM bp_policy_probe")
                out = _j(cur.fetchone()[0])
            assert out is not None and out["policy_identifier"] == "role_definition"
            if "Admin" in details["rules"]["roles"]:
                assert out["rules"]["roles"]["Admin"]["allow"] == ["inspect"]
            with c.cursor() as cur:   # and the rollback takes it back out, still not NULL
                cur.execute(_role_update_sql("2026-10-16-email-settings_rollback.sql")
                            .replace("proc.bp_policy", "bp_policy_probe"))
                cur.execute("SELECT policy_details FROM bp_policy_probe")
                back = _j(cur.fetchone()[0])
            assert back is not None and "SecurityReviewer" not in back["rules"]["roles"]
            assert "inspect" not in back["rules"]["roles"].get("Admin", {}).get("allow", [])
        finally:
            c.rollback()
