"""Apply-on-approve: turn an approved proposal into a versioned active hint.

Approving writes a new active row into proc.bp_prompt
(prompt_type='extraction_vendor_hint'), supersedes any prior active hint for the
same scope (bumping version), marks the proposal approved, and hot-reloads the
in-process hint cache so the next document for that vendor sees it immediately.

Reverting = deactivate the hint row (prompts_status=0); history stays intact.
"""
from __future__ import annotations

import json
import logging

from src.services.agent_actions import PHASE_EXTRACTION, record_action
from src.services.db import get_conn
from src.services.extraction_feedback.hint_store import HINT_STORE

log = logging.getLogger(__name__)


def hint_name(doc_type: str, vendor_key: str, field_name: str | None) -> str:
    return f"vhint::{doc_type}::{vendor_key}::{field_name or '_'}"


def approve(proposal_id: int, approver: str) -> dict:
    """Approve a pending proposal → active versioned bp_prompt hint.

    Returns {"prompt_id": int, "version": int}. Raises ValueError if the
    proposal is missing or not pending.
    """
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT doc_type, vendor_key, field_name, proposed_hint, status "
                "FROM proc.bp_extraction_hint_proposal WHERE proposal_id = %s FOR UPDATE",
                (proposal_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"proposal {proposal_id} not found")
            doc_type, vendor_key, field_name, proposed_hint, status = row
            if status != "pending":
                raise ValueError(f"proposal {proposal_id} is '{status}', not pending")

            name = hint_name(doc_type, vendor_key, field_name)
            # Supersede any current active hint for this exact scope.
            cur.execute(
                "UPDATE proc.bp_prompt SET prompts_status = 0, last_modified_date = now(), "
                "last_modified_by = %s WHERE prompt_name = %s "
                "AND prompt_type = 'extraction_vendor_hint' AND prompts_status = 1",
                (approver, name),
            )
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) FROM proc.bp_prompt "
                "WHERE prompt_name = %s AND prompt_type = 'extraction_vendor_hint'",
                (name,),
            )
            new_version = cur.fetchone()[0] + 1
            desc = json.dumps({
                "scope": {"doc_type": doc_type, "vendor_key": vendor_key, "field_name": field_name},
                "hint_text": proposed_hint,
                "source_proposal_id": proposal_id,
            })
            cur.execute(
                "INSERT INTO proc.bp_prompt "
                "(prompt_name, prompt_type, prompts_desc, prompts_status, version, created_by, last_modified_by) "
                "VALUES (%s, 'extraction_vendor_hint', %s::jsonb, 1, %s, %s, %s) RETURNING prompt_id",
                (name, desc, new_version, approver, approver),
            )
            prompt_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE proc.bp_extraction_hint_proposal SET status = 'approved', "
                "reviewed_by = %s, reviewed_date = now(), resulting_prompt_id = %s "
                "WHERE proposal_id = %s",
                (approver, prompt_id, proposal_id),
            )
        c.commit()

    HINT_STORE.refresh()
    record_action(
        phase=PHASE_EXTRACTION, action_type="hint_approved", agent="extraction_feedback",
        doc_type=doc_type, field_name=field_name,
        summary=f"approved hint proposal {proposal_id} -> prompt {prompt_id} v{new_version}",
        details={"proposal_id": proposal_id, "prompt_id": prompt_id, "vendor_key": vendor_key},
    )
    return {"prompt_id": prompt_id, "version": new_version}


def reject(proposal_id: int, approver: str, reason: str | None = None) -> None:
    """Reject a pending proposal (records the reason; will not be re-proposed)."""
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE proc.bp_extraction_hint_proposal SET status = 'rejected', "
                "reviewed_by = %s, reviewed_date = now(), review_reason = %s "
                "WHERE proposal_id = %s AND status = 'pending'",
                (approver, reason, proposal_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"proposal {proposal_id} not found or not pending")
        c.commit()
    record_action(
        phase=PHASE_EXTRACTION, action_type="hint_rejected", agent="extraction_feedback",
        summary=f"rejected hint proposal {proposal_id}",
        details={"proposal_id": proposal_id, "reason": reason},
    )


def deactivate(prompt_id: int, actor: str) -> None:
    """Revert an active hint (deactivate the bp_prompt row) and hot-reload."""
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE proc.bp_prompt SET prompts_status = 0, last_modified_date = now(), "
                "last_modified_by = %s WHERE prompt_id = %s AND prompt_type = 'extraction_vendor_hint'",
                (actor, prompt_id),
            )
        c.commit()
    HINT_STORE.refresh()
