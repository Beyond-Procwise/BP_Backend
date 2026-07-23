"""Persist / read / update deal-clustering proposals. deal_id is NOT written here —
it is minted only at confirm (Task 11). Callers own the transaction (commit/rollback).
"""
from __future__ import annotations

import json

from src.services.deal_assignment_service import (
    mint_document_id, _persist_deal, _upsert_document_map)
from src.services.agent_actions import record_action, PHASE_CONSOLIDATION


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def store_proposals(cur, batch_deal_id: str, session_id, cluster_result: dict) -> list[int]:
    """Insert every proposal + its members. Atomic within the caller's transaction."""
    ids: list[int] = []
    for prop in cluster_result.get("proposals", []):
        cur.execute(
            "insert into proc.bp_deal_proposal "
            "(batch_deal_id, session_id, proposed_name, confidence, status) "
            "values (%s,%s,%s,%s,'proposed') returning proposal_id",
            (batch_deal_id, session_id, prop["proposed_name"], prop["confidence"]))
        pid = cur.fetchone()[0]
        ids.append(pid)
        for m in prop["members"]:
            cur.execute(
                "insert into proc.bp_deal_proposal_member "
                "(proposal_id, doc_type, doc_pk, base_reference, role, match_score, "
                " match_evidence, review_required, review_reasons) "
                "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (pid, m["doc_type"], str(m["doc_pk"]), m.get("base_reference"), m.get("role"),
                 m.get("match_score"),
                 json.dumps(m.get("match_evidence")) if m.get("match_evidence") is not None else None,
                 prop.get("review_required", False),
                 json.dumps(prop.get("review_reasons") or [])))
    return ids


def list_proposals(cur, batch_deal_id: str) -> list[dict]:
    """Proposals + nested members + evidence, for the review screen."""
    props = _rows(cur,
        "select proposal_id, batch_deal_id, proposed_name, confidence, status, deal_id, "
        "created_at, confirmed_at, confirmed_by from proc.bp_deal_proposal "
        "where batch_deal_id = %s order by confidence desc nulls last, proposal_id",
        (batch_deal_id,))
    for p in props:
        members = _rows(cur,
            "select doc_type, doc_pk, base_reference, role, match_score, match_evidence, "
            "review_required, review_reasons from proc.bp_deal_proposal_member "
            "where proposal_id = %s order by role, doc_pk", (p["proposal_id"],))
        p["members"] = members
    return props


def reject_proposal(cur, proposal_id: int, rejected_by: str) -> None:
    """A rejected grouping is never re-proposed (spec §Human-in-the-loop)."""
    cur.execute(
        "update proc.bp_deal_proposal set status='rejected', "
        "confirmed_at=now(), confirmed_by=%s where proposal_id=%s and status='proposed'",
        (rejected_by, proposal_id))


def update_members(cur, proposal_id: int, remove=None) -> None:
    """Remove members pre-confirm so a reviewer can move/split before accepting."""
    for doc_type, doc_pk in (remove or []):
        cur.execute(
            "delete from proc.bp_deal_proposal_member "
            "where proposal_id=%s and doc_type=%s and doc_pk=%s",
            (proposal_id, doc_type, str(doc_pk)))


class StaleProposalError(Exception):
    """Raised when a proposal's members changed since it was generated (re-extraction).
    Confirm writes nothing and the batch must be re-clustered."""


def mint_proposal_deal_id(batch_deal_id: str, proposal_id: int, primary_supplier=None) -> str:
    """deal_id minted AT CONFIRM (never at proposal time). DEALV3- namespace keeps it
    distinct from the DEALV2-<po> look-back scheme and from the batch label."""
    return f"DEALV3-{proposal_id}"


def confirm_proposal(cur, proposal_id: int, confirmed_by: str, expected_member_pks=None) -> dict:
    """Mint the deal, assign every member document, insert the draft bp_deal row, and
    mark the proposal confirmed. Atomic within the caller's transaction."""
    hdr = _rows(cur, "select proposal_id, batch_deal_id, proposed_name, status "
                     "from proc.bp_deal_proposal where proposal_id=%s", (proposal_id,))
    if not hdr:
        return {"status": "not_found", "proposal_id": proposal_id}
    batch_deal_id = hdr[0].get("batch_deal_id")
    proposed_name = hdr[0].get("proposed_name")

    members = _rows(cur,
        "select doc_type, doc_pk, base_reference, role from proc.bp_deal_proposal_member "
        "where proposal_id=%s", (proposal_id,))
    if expected_member_pks is not None:
        if sorted(str(m["doc_pk"]) for m in members) != sorted(str(p) for p in expected_member_pks):
            raise StaleProposalError(f"proposal {proposal_id} members changed since generation")

    deal_id = mint_proposal_deal_id(batch_deal_id, proposal_id)
    for m in members:
        dt, dpk = m["doc_type"], str(m["doc_pk"])
        doc_id = mint_document_id(deal_id, dt, dpk)
        _persist_deal(cur, dt, dpk, deal_id=deal_id, deal_name=proposed_name,
                      document_id=doc_id, deal_date=None)
        _upsert_document_map(cur, deal_id, proposed_name, dt, dpk, doc_id, None)

    # Draft header — promotion to tracked remains the separate existing gate.
    cur.execute("insert into proc.bp_deal (deal_id, is_tracked) values (%s, false) "
                "on conflict (deal_id) do nothing", (deal_id,))
    cur.execute(
        "update proc.bp_deal_proposal set status='confirmed', deal_id=%s, "
        "confirmed_at=now(), confirmed_by=%s where proposal_id=%s",
        (deal_id, confirmed_by, proposal_id))

    record_action(
        phase=PHASE_CONSOLIDATION, action_type="deal_proposal_confirmed",
        doc_type="deal", doc_pk=deal_id, agent=confirmed_by, status="ok",
        summary=f"confirmed proposal {proposal_id} -> {deal_id} ({len(members)} docs)",
        details={"proposal_id": proposal_id, "batch_deal_id": batch_deal_id,
                 "members": [(m["doc_type"], str(m["doc_pk"])) for m in members]},
        conn=cur.connection)
    return {"status": "confirmed", "proposal_id": proposal_id, "deal_id": deal_id,
            "members": len(members)}
