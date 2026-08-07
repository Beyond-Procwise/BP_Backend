"""bp_approval is the only place a dispatch approval is believed.

The table exists but has never been written to, so this builds both halves.
The tests that matter are the ones proving a near-miss does not count:
right rfq wrong workflow, pending status, and no actioned_by.
"""

import os
import uuid

import psycopg2
import psycopg2.extras
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
    connection.rollback()  # never leave test rows behind
    connection.close()


@pytest.fixture
def ids():
    token = uuid.uuid4().hex[:10]
    return {
        "rfq_id": f"RFQ-{token}",
        "workflow_id": f"WF-{token}",
        "unique_id": f"PROC-WF-{token}",
        "deal_id": f"DEAL-{token}",
    }


@pytest.fixture
def dispatch_policy_id(conn):
    """The bigint key of the dispatch-approval policy Task 1 created."""
    cur = conn.cursor()
    cur.execute(
        "SELECT policy_id FROM proc.bp_policy "
        "WHERE policy_details->>'policy_identifier' = 'email_dispatch_approval' "
        "AND policy_status = 1"
    )
    row = cur.fetchone()
    assert row is not None, "Task 1's dispatch-approval policy is missing"
    return row[0]


def test_recorded_approval_is_found(conn, ids, dispatch_policy_id):
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        deal_id=ids["deal_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        policy_id=dispatch_policy_id,
        policy_name="EmailDispatchApprovalPolicy",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is not None
    assert found["status"] == "approved"
    assert found["actioned_by"] == "buyer@ourcompany.com"
    assert found["deal_id"] == ids["deal_id"]


def test_nothing_recorded_means_nothing_found(conn, ids):
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None


def test_right_rfq_wrong_workflow_does_not_match(conn, ids):
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id="WF-someone-elses",
        unique_id="PROC-WF-someone-elses",
        conn=conn,
    )
    assert found is None


def test_unique_id_only_draft_matches_via_grounding(conn, ids):
    approval_store.record_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is not None


def test_unique_id_only_draft_does_not_match_a_different_unique_id(conn, ids):
    """A draft naming only a unique_id must not match an approval recorded
    for a different unique_id on the same workflow."""
    approval_store.record_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id="PROC-WF-someone-elses-unique-id",
        conn=conn,
    )
    assert found is None


def test_pending_approval_is_not_an_approval(conn, ids):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())",
        (
            ids["rfq_id"],
            ids["workflow_id"],
            "SUP-1",
            "pending",
            "pending",
            "buyer@ourcompany.com",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None


def test_approval_with_no_actioned_by_does_not_count(conn, ids):
    """An approval nobody signed is not a human approval."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,NULL,%s,%s, now())",
        (
            ids["rfq_id"],
            ids["workflow_id"],
            "SUP-1",
            "approved",
            "approved",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None


def test_a_later_revocation_shadows_an_earlier_approval(conn, ids):
    """bp_approval is append-only, so a revocation arrives as a NEW row.

    Filtering the candidate set by status lets the lookup step over the
    revocation and return the superseded approval -- mail keeps going out
    on an authority that was withdrawn, and nothing errors.
    """
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())",
        (
            ids["rfq_id"],
            ids["workflow_id"],
            "SUP-1",
            "deny",
            "revoked",
            "approver@ourcompany.com",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None, "a withdrawn approval must not still authorise a send"


def test_a_later_revocation_shadows_an_earlier_approval_via_grounding(conn, ids):
    """Same shadowing requirement, but for the unique_id-only/grounding path,
    which is a separate query and must not disagree with the rfq_id path."""
    approval_store.record_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())",
        (
            None,
            ids["workflow_id"],
            "SUP-1",
            "deny",
            "revoked",
            "approver@ourcompany.com",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None, "a withdrawn approval must not still authorise a send"


def test_policy_id_is_stored_as_the_numeric_key(conn, ids, dispatch_policy_id):
    """policy_id is a bigint FK-by-convention, not the policy slug."""
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        policy_id=dispatch_policy_id,
        policy_name="EmailDispatchApprovalPolicy",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found["policy_id"] == dispatch_policy_id
    assert found["policy_name"] == "EmailDispatchApprovalPolicy"


def test_deal_id_is_persisted_for_downstream_lookup(conn, ids):
    """Task 7 reads deal_id off the approval row to find competing quotes on
    the same deal; the email draft table has no deal_id column of its own."""
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        deal_id=ids["deal_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found["deal_id"] == ids["deal_id"]
