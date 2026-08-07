"""Read and write proc.bp_approval, the record of who approved what.

The table has the right shape but has never been written to, so dispatch had
nothing to verify against. Both halves live here: the write a human approval
produces, and the lookup the send path trusts.

Only a row that is approved AND carries the name of the person who approved it
counts. An approval nobody signed is not a human approval.

``policy_id`` is the bigint key of ``proc.bp_policy`` (``policy_id`` column),
not the policy slug. ``PolicyEngine`` returns a dict whose ``"policyId"`` key
holds the slug string (e.g. ``"email_dispatch_approval"``) -- that is a
different value from the database column of the same name and must not be
passed here directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import psycopg2.extras

from src.services.db import get_conn

logger = logging.getLogger(__name__)

_STATUS_APPROVED = "approved"

# ApprovalsAgent's verdict vocabulary for the `decision` column (see
# deploy/sql/2026-07-13_bp_approval.sql): approve | require_approval |
# escalate | deny. This is a *different* vocabulary from `status`, which is
# governed separately by policy (Task 1's accepted_status: ["approved"]).
_DECISION_APPROVE = "approve"


def _dict_cursor(conn: Any):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def record_approval(
    *,
    rfq_id: Optional[str],
    workflow_id: Optional[str],
    unique_id: Optional[str],
    supplier_id: Optional[str],
    actioned_by: str,
    deal_id: Optional[str] = None,
    policy_id: Optional[int] = None,  # proc.bp_policy.policy_id, a bigint
    policy_name: Optional[str] = None,  # e.g. "EmailDispatchApprovalPolicy"
    amount: Optional[Any] = None,
    currency: Optional[str] = None,
    conn: Any = None,
) -> int:
    """Record a human approval. Returns the new ``approval_id``."""

    signer = str(actioned_by or "").strip()
    if not signer:
        raise ValueError("actioned_by is required: an approval must name a person")

    grounding = psycopg2.extras.Json({"unique_id": unique_id})
    sql = (
        "INSERT INTO proc.bp_approval "
        "(deal_id, rfq_id, workflow_id, supplier_id, amount, currency, decision, "
        " status, actioned_by, actioned_at, policy_id, policy_name, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now(), %s,%s,%s,%s, now()) "
        "RETURNING approval_id"
    )
    params = (
        deal_id,
        rfq_id,
        workflow_id,
        supplier_id,
        amount,
        currency,
        # decision uses ApprovalsAgent's verdict vocabulary (approve |
        # require_approval | escalate | deny, see
        # deploy/sql/2026-07-13_bp_approval.sql), which is deliberately NOT the
        # same string as status. status='approved' is the governed value
        # Task 1's policy checks (accepted_status: ["approved"]). Do not
        # collapse these back to one value.
        _DECISION_APPROVE,
        _STATUS_APPROVED,
        signer,
        policy_id,
        policy_name,
        grounding,
        signer,
    )

    if conn is not None:
        cur = conn.cursor()
        cur.execute(sql, params)
        return int(cur.fetchone()[0])

    with get_conn() as own:
        own.autocommit = False
        cur = own.cursor()
        try:
            cur.execute(sql, params)
            approval_id = int(cur.fetchone()[0])
            own.commit()
            return approval_id
        except Exception:
            own.rollback()
            raise


def find_dispatch_approval(
    *,
    rfq_id: Optional[str],
    workflow_id: Optional[str],
    unique_id: Optional[str],
    conn: Any = None,
) -> Optional[Dict[str, Any]]:
    """The approval permitting this draft to be sent, or ``None``.

    Matched on rfq_id AND workflow_id. A draft carrying a unique_id but no
    rfq_id matches on workflow_id plus the unique_id recorded in grounding.
    Anything matching neither is absent, not approved.

    bp_approval is append-only: a revocation arrives as a NEW row rather than
    an update to the original. So the newest row for the key is selected
    first, regardless of status, and *that* row is then required to be
    approved and signed. Filtering the candidate set by status before taking
    the newest would let the lookup step over a later revocation and return
    the superseded approval.
    """

    workflow = str(workflow_id or "").strip()
    if not workflow:
        return None

    rfq = str(rfq_id or "").strip()
    unique = str(unique_id or "").strip()

    if rfq:
        sql = (
            "SELECT * FROM ("
            "  SELECT * FROM proc.bp_approval"
            "   WHERE rfq_id = %s AND workflow_id = %s"
            "   ORDER BY approval_id DESC"
            "   LIMIT 1"
            ") latest "
            "WHERE status = %s AND actioned_by IS NOT NULL"
        )
        params: tuple = (rfq, workflow, _STATUS_APPROVED)
    elif unique:
        sql = (
            "SELECT * FROM ("
            "  SELECT * FROM proc.bp_approval"
            "   WHERE workflow_id = %s AND grounding->>'unique_id' = %s"
            "   ORDER BY approval_id DESC"
            "   LIMIT 1"
            ") latest "
            "WHERE status = %s AND actioned_by IS NOT NULL"
        )
        params = (workflow, unique, _STATUS_APPROVED)
    else:
        return None

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = _dict_cursor(connection)
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)


def find_round_approval(
    *,
    workflow_id: Optional[str],
    round_num: Optional[int],
    conn: Any = None,
) -> Optional[Dict[str, Any]]:
    """The approval permitting this negotiation round to be released, or
    ``None``.

    A negotiation round has no rfq_id/unique_id of its own, so this matches
    on workflow_id plus the round number recorded in
    ``grounding->>'round'``. Whoever eventually writes a round approval must
    store the round there, e.g. ``grounding = {"round": round_num}``.

    Same append-only / newest-row-wins-regardless-of-status shape as
    ``find_dispatch_approval``: bp_approval is never updated in place, a
    revocation arrives as a NEW row, so the newest row for the key is
    selected first and *that* row is then required to be approved and
    signed. A caller-supplied decision (e.g. ``hitl_decisions: {"1":
    "approved"}`` in a request payload) is a claim, not an approval, until
    this lookup corroborates it against a row a human actually signed.

    A test double may pass a ``conn`` implementing ``lookup_round_approval``
    directly, in the same style ``email_dispatch_guard`` uses for
    ``lookup_supplier_emails`` etc., so this stays unit-testable without a
    database.
    """

    if hasattr(conn, "lookup_round_approval"):
        return conn.lookup_round_approval(workflow_id=workflow_id, round_num=round_num)

    workflow = str(workflow_id or "").strip()
    if not workflow or round_num is None:
        return None

    sql = (
        "SELECT * FROM ("
        "  SELECT * FROM proc.bp_approval"
        "   WHERE workflow_id = %s AND grounding->>'round' = %s"
        "   ORDER BY approval_id DESC"
        "   LIMIT 1"
        ") latest "
        "WHERE status = %s AND actioned_by IS NOT NULL"
    )
    params = (workflow, str(round_num), _STATUS_APPROVED)

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = _dict_cursor(connection)
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)
