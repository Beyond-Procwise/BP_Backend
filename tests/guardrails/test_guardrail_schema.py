"""The guardrail layer's storage must exist before anything can enforce it.

These assert the deploy SQL has been applied: without the clearance column
every supplier is unclassifiable, and without the six policy rows the gate
has nothing to read and (correctly) denies everything.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()

REQUIRED_POLICIES = {
    "role_definition",
    "role_assignment",
    "email_dispatch_approval",
    "email_recipient_allowlist",
    "email_sensitivity",
    "email_volume",
}


@pytest.fixture(scope="module")
def conn():
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )
    yield connection
    connection.close()


def test_supplier_has_clearance_level_defaulting_to_internal(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_supplier' "
        "AND column_name='clearance_level'"
    )
    row = cur.fetchone()
    assert row is not None, "bp_supplier.clearance_level is missing"
    data_type, is_nullable, default = row
    assert data_type == "text"
    assert is_nullable == "NO"
    assert "internal" in (default or "")


def test_every_existing_supplier_has_a_clearance(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FROM proc.bp_supplier "
        "WHERE clearance_level IS NULL OR clearance_level = ''"
    )
    assert cur.fetchone()[0] == 0


def test_role_assignment_table_exists(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_role_assignment'"
    )
    columns = {r[0] for r in cur.fetchall()}
    assert {
        "assignment_id",
        "subject",
        "role",
        "granted_by",
        "granted_at",
        "revoked_at",
    } <= columns


def test_six_guardrail_policies_are_active(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT policy_details->>'policy_identifier' FROM proc.bp_policy "
        "WHERE policy_status = 1"
    )
    identifiers = {r[0] for r in cur.fetchall() if r[0]}
    missing = REQUIRED_POLICIES - identifiers
    assert not missing, f"missing guardrail policies: {sorted(missing)}"


def test_guardrail_policies_declare_a_required_role(conn):
    """The RBAC ruling is a first-class key, not an afterthought."""
    cur = conn.cursor()
    cur.execute(
        "SELECT policy_details->>'policy_identifier', policy_details->>'required_role' "
        "FROM proc.bp_policy WHERE policy_status = 1 "
        "AND policy_details->>'policy_identifier' = ANY(%s)",
        (sorted(REQUIRED_POLICIES),),
    )
    for identifier, required_role in cur.fetchall():
        assert required_role in {"Viewer", "Buyer", "Approver", "Admin"}, (
            f"{identifier} has no usable required_role"
        )
