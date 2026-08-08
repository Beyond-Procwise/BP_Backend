"""I5: prove a denied send's audit row survives the caller's own rollback.

Task 5 proved the strict writer persists on its own connection. Task 7
proved the five checks with fakes. Nothing drove ``send_draft`` itself to a
denial and then read the audit row back on a FRESH connection -- and that
gap is structurally why C2 survived nine reviews: ``record_action_or_fail``
was called with ``conn=conn`` (the caller's own, shared connection), which
writes inside a SAVEPOINT and does not commit. ``DispatchDenied`` then
propagates out of ``with self.agent_nick.get_db_connection() as conn:``,
which rolls back on exception -- taking the unset audit row with it. A
fresh connection then finds nothing: the deny half of spec S5.6/G8, the half
that matters for detecting an attack, is invisible.

This test uses REAL, independent psycopg2 connections (not the in-memory
fakes the rest of the send-path suite uses) because the bug is specifically
about transaction boundaries between two connections, which an in-memory
double cannot reproduce faithfully.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import psycopg2
import pytest
from dotenv import load_dotenv

from src.services import email_dispatch_guard, guardrail
from src.services.email_dispatch_service import EmailDispatchService

load_dotenv()


def _pg_kwargs():
    return dict(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )


def _connect():
    return psycopg2.connect(**_pg_kwargs())


@pytest.fixture(scope="module")
def _db_reachable():
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"bp_sqldb not reachable: {exc}")
    conn.close()


@pytest.fixture
def draft_row(_db_reachable):
    """A real, minimal draft row on its own committed connection, so
    send_draft's own (separate) connection can see it. Deleted
    unconditionally in teardown, along with any audit row this test
    produced -- nothing is left behind on the shared cluster.
    """
    unique_id = f"PROC-WF-AUDIT-SURVIVES-{uuid.uuid4().hex[:10]}"
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO proc.draft_rfq_emails "
            "(rfq_id, subject, body, recipient_email, unique_id, sent) "
            "VALUES (%s,%s,%s,%s,%s,false)",
            (unique_id, "Test subject", "Test body", "buyer@example.com", unique_id),
        )
        conn.commit()
    finally:
        conn.close()

    yield unique_id

    cleanup = _connect()
    try:
        cur = cleanup.cursor()
        cur.execute(
            "DELETE FROM proc.draft_rfq_emails WHERE unique_id = %s", (unique_id,)
        )
        cur.execute(
            "DELETE FROM proc.bp_agent_actions WHERE details ->> 'unique_id' = %s",
            (unique_id,),
        )
        cleanup.commit()
    finally:
        cleanup.close()


class _FakePrincipal:
    subject = "sub-audit-survives"
    claims = {"cognito:groups": ["bp-approvers"]}


class _RealConnAgentNick:
    """Opens a genuine, independent psycopg2 connection on every call --
    exactly what production's EmailDispatchService does
    (``self.agent_nick.get_db_connection()``). Two calls give two REAL,
    independent transactions, which is the property this test needs: the
    outer one rolls back on the raised DispatchDenied, and the audit write
    must not be sitting on it.
    """

    settings = SimpleNamespace(ses_default_sender="sender@example.com")

    def get_db_connection(self):
        return _connect()


def test_a_denied_sends_audit_row_survives_the_callers_own_rollback(
    monkeypatch, draft_row
):
    """RED against the pre-C2-fix code: record_action_or_fail(conn=conn)
    writes inside a SAVEPOINT on send_draft's own connection. DispatchDenied
    then propagates out of `with self.agent_nick.get_db_connection() as
    conn:`, which rolls back on exception -- taking the audit row with it. A
    fresh connection then finds nothing.
    """
    denial = guardrail.Decision(
        allowed=False,
        reason="test forced denial for I5",
        policy_name="EmailDispatchApprovalPolicy",
    )
    monkeypatch.setattr(email_dispatch_guard, "check_dispatch", lambda **_: denial)

    service = EmailDispatchService(_RealConnAgentNick())

    with pytest.raises(email_dispatch_guard.DispatchDenied):
        service.send_draft(draft_row, principal=_FakePrincipal())

    fresh = _connect()
    try:
        cur = fresh.cursor()
        cur.execute(
            "SELECT status, summary FROM proc.bp_agent_actions "
            "WHERE action_type = 'email.send' AND details ->> 'unique_id' = %s",
            (draft_row,),
        )
        row = cur.fetchone()
    finally:
        fresh.rollback()
        fresh.close()

    assert row is not None, (
        "the denial's audit row did not survive on a fresh connection -- it "
        "was written on the caller's own transaction and rolled back with it"
    )
    assert row[0] == "denied"
    assert row[1] == denial.reason
