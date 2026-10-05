"""A goods receipt reaches _trgt by its purchase order's deal, or not at all.

A receipt's whole value is that it attaches to the order. Standing alone it says
"ten of something arrived"; attached to a PO line it says what was ordered, what
arrived and -- once an invoice joins them -- whether more was billed than
received.

So the rule here is deliberately narrow. The receipt cites a PO; that PO is
resolved with `linking_engine._pick_po`, the SAME function the two-way match
uses, so "PO-4500018832", "4500018832" and "4500018832 (Rev 1)" all reach the
same order. If the PO is not held, or is held but has not been grouped into a
deal yet, the receipt STAYS in `_stg` and says why. It does not land deal-less
in `_trgt`, where every reader keys on `deal_id` and would simply never see it,
and it does not invent a deal of its own.

This is the only goods-receipt-specific step in the whole ingestion path.
Everything before it -- parse, extract, `_raw`, `_stg` -- runs through the
ordinary writers, and everything after it reads `_trgt` like any other document.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

DOC_TYPE = "goods_receipt"

#: The PO columns a receipt inherits. `expected_delivery_date` is what
#: deal_assignment_service.resolve_deal_date treats as the deal's date, so the
#: receipt gets the same answer the invoice on that deal got.
_PO_COLS = "t.po_id, t.deal_id, t.deal_name, t.expected_delivery_date"


def link_receipt_to_po(grn_id: str) -> dict[str, Any]:
    """Attach one staged goods receipt to its purchase order's deal.

    Returns ``{"linked": bool, "reason": str | None, "deal_id": str | None,
    "po_id": str | None}``. Never raises for an absent PO or an ungrouped one --
    those are ordinary states of a real corpus, reported rather than failed.
    """
    from src.services.deal_assignment_service import (
        _ensure_in_trgt, _persist_deal, mint_document_id, resolve_deal_date,
    )
    from src.services.linking_engine import _pick_po

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT po_id FROM proc.bp_goods_receipt_stg WHERE grn_id = %s",
                (grn_id,),
            )
            staged = cur.fetchone()
            if staged is None:
                conn.rollback()
                return _no(grn_id, "receipt_not_staged")
            cited_po = staged[0]
            if not (cited_po or "").strip():
                conn.rollback()
                return _no(grn_id, "receipt_cites_no_po")

            po = _pick_po(cur, cited_po, cols=_PO_COLS)
            if po is None:
                conn.rollback()
                return _no(grn_id, "no_matching_po", po_id=cited_po)

            deal_id = (po.get("deal_id") or "").strip()
            if not deal_id:
                conn.rollback()
                return _no(grn_id, "po_has_no_deal", po_id=po.get("po_id"))

            # The receipt now records the PO it actually resolved to, not the
            # spelling the document used, so every downstream join on po_id --
            # the three-way match included -- sees one canonical value.
            cur.execute(
                "UPDATE proc.bp_goods_receipt_stg SET po_id = %s WHERE grn_id = %s",
                (po.get("po_id"), grn_id))
            cur.execute(
                "UPDATE proc.bp_goods_receipt_line_items_stg SET po_id = %s "
                "WHERE grn_id = %s", (po.get("po_id"), grn_id))

            if not _ensure_in_trgt(cur, DOC_TYPE, grn_id):
                conn.rollback()
                return _no(grn_id, "trgt_copy_failed", po_id=po.get("po_id"))

            _persist_deal(
                cur, DOC_TYPE, grn_id,
                deal_id=deal_id,
                deal_name=po.get("deal_name"),
                document_id=mint_document_id(deal_id, DOC_TYPE, str(grn_id)),
                deal_date=resolve_deal_date(po),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    log.info("goods receipt %s linked to PO %s on deal %s",
             grn_id, po.get("po_id"), deal_id)
    return {"linked": True, "reason": None, "deal_id": deal_id,
            "po_id": po.get("po_id")}


def _no(grn_id: str, reason: str, *, po_id: Optional[str] = None) -> dict[str, Any]:
    log.info("goods receipt %s not linked: %s (po=%s)", grn_id, reason, po_id)
    return {"linked": False, "reason": reason, "deal_id": None, "po_id": po_id}
