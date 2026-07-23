"""Deal clustering proposals: generate per batch, review, confirm, adjust.
Nothing here auto-applies a deal — confirm is the only write that mints deal_id.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


def _fetch_batch(cur, batch_deal_id: str) -> dict:
    """Read the batch's extracted rows for clustering. Uses process_monitor.deal_id as the
    batch label (see upload-path change, Task 13). Returns the cluster_batch kwargs."""
    from src.services.linking_engine import _rows
    # documents whose process_monitor batch label == batch_deal_id, joined to _stg rows.
    quotes = _rows(cur,
        "select q.* from proc.bp_quote_stg q join proc.bp_quote_raw r on r.quote_id=q.quote_id "
        "join proc.process_monitor pm on pm.id=r.process_monitor_id where pm.deal_id=%s",
        (batch_deal_id,))
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


def _generate(batch_deal_id: str, session_id: Optional[str]) -> dict:
    """Cluster a batch and persist proposals. Atomic: commit or rollback."""
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            kwargs = _fetch_batch(cur, batch_deal_id)
            declared = declared_groups(cur, batch_deal_id)
            result = deal_clustering.cluster_batch(declared=declared, **kwargs)
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
def generate(body: GenerateBody) -> dict[str, Any]:
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
def confirm(proposal_id: int, body: ConfirmBody) -> dict[str, Any]:
    try:
        return _confirm(proposal_id, body.confirmed_by, body.expected_member_pks)
    except proposal_store.StaleProposalError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.patch("/proposals/{proposal_id}/members", summary="Move/remove documents pre-confirm")
def patch_members(proposal_id: int, body: MembersBody) -> dict[str, Any]:
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
