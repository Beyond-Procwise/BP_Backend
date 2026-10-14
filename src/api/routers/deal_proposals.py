"""Deal clustering proposals: generate per batch, review, confirm, adjust.
Nothing here auto-applies a deal — confirm is the only write that mints deal_id.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from src.services.db import get_conn
from src.services import deal_clustering, proposal_store
from src.services.declared_linkage import declared_groups

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/deals", tags=["Deals"])


class GenerateBody(BaseModel):
    batch_deal_id: str
    session_id: Optional[str] = None


class ConfirmBody(BaseModel):
    confirmed_by: str
    expected_member_pks: Optional[list[str]] = None


class MembersBody(BaseModel):
    remove: list[list[str]] = []


class RejectBody(BaseModel):
    rejected_by: str


def _attach_rfq_refs(cur, quotes: list[dict]) -> None:
    """Enrich each quote with the RFQ reference its parsed text cites (tier-1
    linked identifier for rivalry clustering). Reads the raw tier's stored
    parser snapshot — no re-parse, deterministic regex only."""
    from src.services.deal_clustering import extract_rfq_reference
    if not quotes:
        return
    cur.execute(
        "select quote_id, parser_snapshot->>'full_text' from proc.bp_quote_raw "
        "where quote_id = any(%s) order by raw_id",
        ([q["quote_id"] for q in quotes],))
    texts = {qid: txt for qid, txt in cur.fetchall()}  # later raws win
    for q in quotes:
        q["rfq_reference"] = extract_rfq_reference(texts.get(q["quote_id"]) or "")


def _fetch_batch(cur, batch_deal_id: str) -> dict:
    """Read the batch's extracted rows for clustering. Uses process_monitor.deal_id as the
    batch label (see upload-path change, Task 13). Returns the cluster_batch kwargs."""
    from src.services.linking_engine import _rows
    # documents whose process_monitor batch label == batch_deal_id, joined to _stg rows.
    quotes = _rows(cur,
        "select q.* from proc.bp_quote_stg q join proc.bp_quote_raw r on r.quote_id=q.quote_id "
        "join proc.process_monitor pm on pm.id=r.process_monitor_id where pm.deal_id=%s",
        (batch_deal_id,))
    _attach_rfq_refs(cur, quotes)
    quote_lines: dict = {}
    for q in quotes:
        quote_lines[q["quote_id"]] = _rows(cur,
            "select * from proc.bp_quote_line_items_stg where quote_id=%s", (q["quote_id"],))
    pos = _rows(cur,
        "select p.* from proc.bp_purchase_order_stg p join proc.bp_purchase_order_raw r "
        "on r.po_id=p.po_id join proc.process_monitor pm on pm.id=r.process_monitor_id "
        "where pm.deal_id=%s", (batch_deal_id,))
    po_lines = {p["po_id"]: _rows(cur, "select * from proc.bp_po_line_items_stg where po_id=%s",
                                  (p["po_id"],)) for p in pos}
    invoices = _rows(cur,
        "select i.* from proc.bp_invoice_stg i join proc.bp_invoice_raw r on r.invoice_id=i.invoice_id "
        "join proc.process_monitor pm on pm.id=r.process_monitor_id where pm.deal_id=%s",
        (batch_deal_id,))
    return {"quotes": quotes, "quote_lines": quote_lines, "purchase_orders": pos,
            "po_lines": po_lines, "invoices": invoices}


def _drop_rejected_pairings(cur, batch_deal_id: str, proposals: list[dict]) -> list[dict]:
    """A rejected pairing must never resurface (spec §Human-in-the-loop: "A confirmed
    grouping is never re-proposed — including a rejected one"). Compares on the set of
    quote doc_pks (the bids that define the pairing), not the full member set (which
    also carries po/invoice)."""
    rejected_sets = set(proposal_store.rejected_member_sets(cur, batch_deal_id))
    if not rejected_sets:
        return proposals
    kept = []
    for prop in proposals:
        quote_pks = frozenset(str(m["doc_pk"]) for m in prop.get("members", [])
                              if m.get("doc_type") == "quote")
        if quote_pks and quote_pks in rejected_sets:
            continue
        kept.append(prop)
    return kept


def _generate(batch_deal_id: str, session_id: Optional[str]) -> dict:
    """Cluster a batch and persist proposals. Atomic: commit or rollback."""
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            kwargs = _fetch_batch(cur, batch_deal_id)
            declared = declared_groups(cur, batch_deal_id)
            result = deal_clustering.cluster_batch(declared=declared, **kwargs)
            result["proposals"] = _drop_rejected_pairings(cur, batch_deal_id, result["proposals"])
            # Regenerate REPLACES the prior un-actioned ('proposed') set instead of
            # stacking duplicates; confirmed/rejected/superseded rows are preserved.
            proposal_store.delete_proposed(cur, batch_deal_id)
            ids = proposal_store.store_proposals(cur, batch_deal_id, session_id, result)
            conn.commit()
            return {"batch_deal_id": batch_deal_id, "proposal_ids": ids,
                    "ungrouped": result["ungrouped"],
                    "members_with_lines": result["members_with_lines"],
                    "members_total": result["members_total"]}
        except Exception:
            conn.rollback()
            raise


def _confirm(proposal_id: int, confirmed_by: str, expected: Optional[list]) -> dict:
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            out = proposal_store.confirm_proposal(cur, proposal_id, confirmed_by, expected)
            conn.commit()
            return out
        except Exception:
            conn.rollback()
            raise


@router.post("/proposals/generate", summary="Cluster an upload batch into proposed deals")
def generate(body: GenerateBody, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        return _generate(body.batch_deal_id, body.session_id)
    except Exception as exc:  # atomic failure — no partial proposals (spec §Error handling)
        logger.exception("proposal generation failed for %s", body.batch_deal_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/proposals", summary="List proposals + members + evidence for a batch")
def list_(batch: str) -> dict[str, Any]:
    with get_conn() as conn:
        return {"batch_deal_id": batch, "proposals": proposal_store.list_proposals(conn.cursor(), batch)}


@router.post("/proposals/{proposal_id}/confirm", summary="Confirm a proposal — mints the deal")
def confirm(proposal_id: int, body: ConfirmBody,
            principal=Depends(require_user)) -> dict[str, Any]:
    # Confirming mints the deal, and the name on it lands in
    # bp_deal_proposal.confirmed_by AND bp_agent_actions.agent. That name is
    # the token's. `body.confirmed_by` stays on the model because clients send
    # it; it is not read, and with no principal the deal is confirmed by nobody.
    try:
        return _confirm(proposal_id, getattr(principal, "subject", None) or None,
                        body.expected_member_pks)
    except proposal_store.StaleProposalError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/proposals/{proposal_id}/reject", summary="Reject a proposal — never re-proposed")
def reject(proposal_id: int, body: RejectBody,
           principal=Depends(require_user)) -> dict[str, Any]:
    # A rejected grouping is never re-proposed, so who rejected it matters as
    # much as who confirmed one. The token, not `body.rejected_by`.
    with get_conn() as conn:
        conn.autocommit = False
        try:
            proposal_store.reject_proposal(conn.cursor(), proposal_id,
                                           getattr(principal, "subject", None) or None)
            conn.commit()
            return {"status": "rejected", "proposal_id": proposal_id}
        except Exception:
            conn.rollback()
            raise


@router.patch("/proposals/{proposal_id}/members", summary="Move/remove documents pre-confirm")
def patch_members(proposal_id: int, body: MembersBody,
                  principal=Depends(require_user)) -> dict[str, Any]:
    with get_conn() as conn:
        conn.autocommit = False
        try:
            proposal_store.update_members(conn.cursor(), proposal_id,
                                          remove=[(t, p) for t, p in body.remove])
            conn.commit()
            return {"status": "ok", "proposal_id": proposal_id}
        except Exception:
            conn.rollback()
            raise
