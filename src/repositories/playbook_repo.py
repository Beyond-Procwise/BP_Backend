"""Playbooks, under their authorship lifecycle.

    draft -> pending_approval -> active -> retired

Only ``active`` playbooks are loaded by the store, and therefore only an
approved strategy can propose anything.

The two rules that make an approval mean something are pure functions at the
top of this module, so the router cannot hold a different opinion about them:

  * Editing an ``active`` playbook returns it to ``pending_approval`` and bumps
    ``version``. An approved strategy cannot be changed underneath its approval.
  * Nobody approves their own work, and an unattributable approval is refused
    outright -- without a subject the bar cannot be applied at all.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.db import get_conn
from services.playbooks.finding_source import validate_trigger_match

logger = logging.getLogger(__name__)


class LifecycleError(ValueError):
    """A transition the lifecycle does not allow. The router turns this into a 400."""


_EDITABLE = {
    "draft": ("draft", False),
    "pending_approval": ("pending_approval", False),
    # An approved strategy cannot be changed underneath its approval.
    "active": ("pending_approval", True),
}


def next_status_for_edit(current: str) -> Tuple[str, bool]:
    """``(status after an edit, whether to bump version)``."""

    try:
        return _EDITABLE[str(current)]
    except KeyError:
        raise LifecycleError(
            f"a {current} playbook cannot be edited. Copy it into a new draft "
            "instead -- a retired strategy is kept so the proposals it raised "
            "stay legible."
        ) from None


def check_approval(
    *,
    authored_by: str,
    approver: Optional[str],
    current_status: str,
    workflow_is_active: bool,
) -> None:
    """Raise unless ``approver`` may move this playbook to ``active``."""

    subject = (approver or "").strip()
    if not subject:
        raise LifecycleError(
            "an approval must name who gave it: without a subject the "
            "self-approval bar cannot be applied at all."
        )
    if current_status != "pending_approval":
        raise LifecycleError(
            f"only a pending_approval playbook can be approved; this one is "
            f"{current_status}."
        )
    if subject.casefold() == (authored_by or "").strip().casefold():
        raise LifecycleError(
            "a playbook cannot be approved by its own author. Ask someone else "
            "to review it."
        )
    if not workflow_is_active:
        raise LifecycleError(
            "the workflow this playbook points at is missing or inactive. "
            "Approving it would create a strategy that can only fail at the "
            "moment somebody accepts its proposal."
        )


_COLUMNS = (
    "playbook_id, playbook_name, description, trigger_source, trigger_match, "
    "agent_workflow_id, params, playbook_status, version, authored_by, "
    "approved_by, approved_at, created_at, last_modified_at, last_modified_by"
)


def _row(r) -> Dict[str, Any]:
    def _obj(value):
        return json.loads(value) if isinstance(value, str) else (value or {})

    return {
        "playbook_id": r[0], "playbook_name": r[1], "description": r[2],
        "trigger_source": r[3], "trigger_match": _obj(r[4]),
        "agent_workflow_id": r[5], "params": _obj(r[6]),
        "playbook_status": r[7], "version": r[8], "authored_by": r[9],
        "approved_by": r[10],
        "approved_at": r[11].isoformat() if r[11] else None,
        "created_at": r[12].isoformat() if r[12] else None,
        "last_modified_at": r[13].isoformat() if r[13] else None,
        "last_modified_by": r[14],
    }


def get(playbook_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_playbook WHERE playbook_id = %s",
                (playbook_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return _row(row) if row else None


def list_playbooks(status: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = f"SELECT {_COLUMNS} FROM proc.bp_playbook"
    params: tuple = ()
    if status:
        sql += " WHERE playbook_status = %s"
        params = (status,)
    sql += " ORDER BY playbook_id DESC"
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            cur.close()
    return [_row(r) for r in rows]


def create(
    *,
    name: str,
    trigger_source: str,
    trigger_match: Dict[str, Any],
    agent_workflow_id: int,
    params: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    authored_by: str,
) -> int:
    """Insert a draft. Raises ``ValueError`` on an unknown match key."""

    match = validate_trigger_match(trigger_source, trigger_match or {})
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO proc.bp_playbook "
                "(playbook_name, description, trigger_source, trigger_match, "
                " agent_workflow_id, params, playbook_status, authored_by, "
                " last_modified_by) "
                "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, 'draft', %s, %s) "
                "RETURNING playbook_id",
                (name, description, trigger_source, json.dumps(match),
                 agent_workflow_id, json.dumps(params or {}),
                 authored_by, authored_by),
            )
            return int(cur.fetchone()[0])
        finally:
            cur.close()


def update(
    playbook_id: int,
    *,
    name: str,
    trigger_source: str,
    trigger_match: Dict[str, Any],
    agent_workflow_id: int,
    params: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    modified_by: str,
) -> Dict[str, Any]:
    """Edit a playbook. An active one returns to pending_approval, version + 1."""

    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    status, bump = next_status_for_edit(existing["playbook_status"])
    match = validate_trigger_match(trigger_source, trigger_match or {})
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET "
                "  playbook_name = %s, description = %s, trigger_source = %s, "
                "  trigger_match = %s::jsonb, agent_workflow_id = %s, "
                "  params = %s::jsonb, playbook_status = %s, "
                "  version = version + %s, "
                # An edit unapproves. Leaving approved_by set would leave a row
                # that says somebody signed off on text they never saw.
                "  approved_by = CASE WHEN %s THEN NULL ELSE approved_by END, "
                "  approved_at = CASE WHEN %s THEN NULL ELSE approved_at END, "
                "  last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (name, description, trigger_source, json.dumps(match),
                 agent_workflow_id, json.dumps(params or {}), status,
                 1 if bump else 0, bump, bump, modified_by, playbook_id),
            )
        finally:
            cur.close()
    return get(playbook_id)


def submit(playbook_id: int, *, modified_by: str) -> None:
    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    if existing["playbook_status"] not in ("draft", "pending_approval"):
        raise LifecycleError(
            f"a {existing['playbook_status']} playbook cannot be submitted for approval"
        )
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'pending_approval', "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (modified_by, playbook_id),
            )
        finally:
            cur.close()


def workflow_is_active(agent_workflow_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT is_active FROM proc.bp_agent_workflow WHERE workflow_id = %s",
                (agent_workflow_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return bool(row and row[0])


def approve(playbook_id: int, *, approver: str) -> Dict[str, Any]:
    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    check_approval(
        authored_by=existing["authored_by"],
        approver=approver,
        current_status=existing["playbook_status"],
        workflow_is_active=workflow_is_active(existing["agent_workflow_id"]),
    )
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'active', "
                "approved_by = %s, approved_at = now(), "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (approver, approver, playbook_id),
            )
        finally:
            cur.close()
    return get(playbook_id)


def retire(playbook_id: int, *, modified_by: str) -> None:
    """Stop a playbook proposing. Its existing proposals stay decidable."""

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'retired', "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (modified_by, playbook_id),
            )
        finally:
            cur.close()


# -- proposals -----------------------------------------------------------

_PROPOSAL_COLUMNS = (
    "p.proposal_id, p.playbook_id, p.finding_source, p.finding_id, p.deal_id, "
    "p.proposal_status, p.evidence, p.run_id, p.proposed_at, p.decided_by, "
    "p.decided_at, p.decision_reason, b.playbook_name, b.agent_workflow_id, b.params"
)


def _proposal_row(r) -> Dict[str, Any]:
    def _obj(value):
        return json.loads(value) if isinstance(value, str) else (value or {})

    return {
        "proposal_id": r[0], "playbook_id": r[1], "finding_source": r[2],
        "finding_id": r[3], "deal_id": r[4], "proposal_status": r[5],
        "evidence": _obj(r[6]), "run_id": r[7],
        "proposed_at": r[8].isoformat() if r[8] else None,
        "decided_by": r[9],
        "decided_at": r[10].isoformat() if r[10] else None,
        "decision_reason": r[11], "playbook_name": r[12],
        "agent_workflow_id": r[13], "params": _obj(r[14]),
    }


def get_proposal(proposal_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM proc.bp_playbook_proposal p "
                "JOIN proc.bp_playbook b ON b.playbook_id = p.playbook_id "
                "WHERE p.proposal_id = %s",
                (proposal_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return _proposal_row(row) if row else None


def list_proposals(status: Optional[str] = "proposed", limit: int = 100) -> List[Dict[str, Any]]:
    sql = (
        f"SELECT {_PROPOSAL_COLUMNS} FROM proc.bp_playbook_proposal p "
        "JOIN proc.bp_playbook b ON b.playbook_id = p.playbook_id"
    )
    params: tuple = ()
    if status:
        sql += " WHERE p.proposal_status = %s"
        params = (status,)
    sql += " ORDER BY p.proposed_at DESC LIMIT %s"
    params = params + (max(1, min(int(limit), 1000)),)
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            cur.close()
    return [_proposal_row(r) for r in rows]


def _decide(proposal_id: int, status: str, by: str,
            reason: Optional[str] = None, run_id: Optional[str] = None) -> None:
    """Record a decision, once.

    The WHERE clause pins proposal_status = 'proposed' so two people clicking
    approve at the same moment cannot both write a decision -- only the first
    UPDATE matches.
    """

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook_proposal "
                "   SET proposal_status = %s, decided_by = %s, decided_at = now(), "
                "       decision_reason = COALESCE(%s, decision_reason), "
                "       run_id = COALESCE(%s, run_id) "
                " WHERE proposal_id = %s AND proposal_status = 'proposed'",
                (status, by, reason, run_id, proposal_id),
            )
        finally:
            cur.close()


def mark_proposal_executed(proposal_id: int, run_id: Optional[str], by: str) -> None:
    _decide(proposal_id, "executed", by, run_id=run_id)


def mark_proposal_rejected(proposal_id: int, by: str, reason: str) -> None:
    _decide(proposal_id, "rejected", by, reason=reason)


def mark_proposal_superseded(proposal_id: int, by: str) -> None:
    _decide(proposal_id, "superseded", by,
            reason="the finding this was raised for is no longer open")
