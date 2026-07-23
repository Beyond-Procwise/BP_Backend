"""Persist / read / update deal-clustering proposals. deal_id is NOT written here —
it is minted only at confirm (Task 11). Callers own the transaction (commit/rollback).
"""
from __future__ import annotations

import json


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
