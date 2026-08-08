"""Listing what awaits approval, and withdrawing one that was given.

Revocation writes a NEW row rather than mutating: bp_approval is append-only,
and find_dispatch_approval already takes the newest row for a key regardless
of status, so a later revoked row shadows the approval.
"""

import os
import uuid

import psycopg2
import pytest
from dotenv import load_dotenv

from src.services import approval_store

load_dotenv()


@pytest.fixture
def conn():
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )
    connection.autocommit = False
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def ids():
    token = uuid.uuid4().hex[:10]
    return {
        "rfq_id": f"RFQ-{token}",
        "workflow_id": f"WF-{token}",
        "unique_id": f"PROC-WF-{token}",
    }


def test_revoking_shadows_the_approval(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    assert approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], conn=conn,
    ) is not None

    approval_store.revoke_approval(
        approval_id=approval_id, actioned_by="approver@ourcompany.com",
        reason="sent in error", conn=conn,
    )

    assert approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], conn=conn,
    ) is None, "a withdrawn approval must not still authorise a send"


def test_revoking_records_who_withdrew_it(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    new_id = approval_store.revoke_approval(
        approval_id=approval_id, actioned_by="approver@ourcompany.com",
        reason="sent in error", conn=conn,
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT status, actioned_by, decision_reason FROM proc.bp_approval "
        "WHERE approval_id = %s",
        (new_id,),
    )
    status, actioned_by, reason = cur.fetchone()
    assert status == "revoked"
    assert actioned_by == "approver@ourcompany.com"
    assert reason == "sent in error"


def test_revoking_an_unknown_approval_raises(conn):
    with pytest.raises(ValueError):
        approval_store.revoke_approval(
            approval_id=-1, actioned_by="approver@ourcompany.com", conn=conn
        )


def test_revocation_requires_a_named_person(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    with pytest.raises(ValueError):
        approval_store.revoke_approval(
            approval_id=approval_id, actioned_by="   ", conn=conn
        )


def test_an_approved_draft_is_not_pending(conn, ids):
    """Insert a draft, approve it, and confirm it drops out of the list."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.draft_rfq_emails "
        "(rfq_id, supplier_id, subject, body, sent, recipient_email, "
        " thread_index, sender, workflow_id, unique_id, created_on) "
        "VALUES (%s,%s,%s,%s,false,%s,1,%s,%s,%s, now())",
        (ids["rfq_id"], "SUP-1", "RFQ", "Please quote.",
         "buyer@supplier-b.com", "us@ourcompany.com",
         ids["workflow_id"], ids["unique_id"]),
    )
    pending = approval_store.list_pending_dispatch_approvals(limit=200, conn=conn)
    assert any(r["unique_id"] == ids["unique_id"] for r in pending)

    approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    pending_after = approval_store.list_pending_dispatch_approvals(limit=200, conn=conn)
    assert not any(r["unique_id"] == ids["unique_id"] for r in pending_after)


# ---------------------------------------------------------------------------
# I7 -- negotiation_workflow_exists, the lookup approve_round was missing.
# ---------------------------------------------------------------------------


def test_negotiation_workflow_exists_is_false_for_an_unknown_workflow(conn, ids):
    assert approval_store.negotiation_workflow_exists(
        workflow_id=ids["workflow_id"], conn=conn
    ) is False


def test_negotiation_workflow_exists_is_true_once_a_session_state_row_exists(conn, ids):
    """negotiation_session_state is the table actually written to (live
    counts at the time this was added: 7 rows there vs 1 in
    negotiation_sessions)."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.negotiation_session_state (workflow_id, supplier_id) "
        "VALUES (%s, %s)",
        (ids["workflow_id"], "SUP-1"),
    )
    assert approval_store.negotiation_workflow_exists(
        workflow_id=ids["workflow_id"], conn=conn
    ) is True


def test_negotiation_workflow_exists_is_true_for_a_negotiation_sessions_row(conn, ids):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.negotiation_sessions (workflow_id, supplier_id, round) "
        "VALUES (%s, %s, %s)",
        (ids["workflow_id"], "SUP-1", 1),
    )
    assert approval_store.negotiation_workflow_exists(
        workflow_id=ids["workflow_id"], conn=conn
    ) is True


def test_negotiation_workflow_exists_returns_false_for_a_blank_workflow_id(conn):
    assert approval_store.negotiation_workflow_exists(workflow_id="", conn=conn) is False
    assert approval_store.negotiation_workflow_exists(workflow_id=None, conn=conn) is False
