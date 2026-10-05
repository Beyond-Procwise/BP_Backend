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


def link_pending_receipts(limit: int = 500) -> dict[str, Any]:
    """Retry every staged goods receipt that has not reached _trgt yet.

    `link_receipt_to_po` is one best-effort call at ingestion. A note that
    arrives BEFORE its purchase order is extracted, or before that order is
    grouped into a deal, reports `no_matching_po` or `po_has_no_deal` and then
    waits — and nothing was waking it. Found in the whole-branch review of
    2026-10-05: `goods_receipt` was added to `deal_assignment_service._DOC` so
    the generic helpers would apply to it, but every sweep in that module
    iterates a hard-coded `("invoice", "quote", "po")`, so no sweep ever reached
    a receipt. The note stayed in `_stg` for ever, invisible to the match and to
    every reader.

    Idempotent: a receipt already in `_trgt` is skipped, and a receipt whose PO
    still has no deal simply reports that again. Returns counts, so the
    scheduler can log whether anything moved.
    """
    out: dict[str, Any] = {"linked": 0, "waiting": 0, "by_reason": {}}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT s.grn_id FROM proc.bp_goods_receipt_stg s "
            " WHERE NOT EXISTS (SELECT 1 FROM proc.bp_goods_receipt_trgt t "
            "                    WHERE t.grn_id = s.grn_id) "
            " ORDER BY s.grn_id LIMIT %s", (limit,))
        pending = [r[0] for r in cur.fetchall()]

    for grn_id in pending:
        try:
            result = link_receipt_to_po(grn_id)
        except Exception:  # noqa: BLE001 - one bad row must not stop the sweep
            log.exception("goods receipt %s: link retry failed", grn_id)
            out["by_reason"]["error"] = out["by_reason"].get("error", 0) + 1
            continue
        if result["linked"]:
            out["linked"] += 1
        else:
            out["waiting"] += 1
            reason = result["reason"] or "unknown"
            out["by_reason"][reason] = out["by_reason"].get(reason, 0) + 1
    return out


def run_match_for_receipt(grn_id: str) -> dict[str, Any]:
    """Run the three-way match for a receipt that is already persisted.

    The match has to happen AFTER the receipt reaches `_stg` and `_trgt`, for
    two reasons found on the live re-run of 2026-10-05. The counts it records
    are an UPDATE on those rows, so run earlier it silently wrote nothing and
    the deal then read NOT ASSESSED although every line had been compared --
    the verdict gate turned from a fix into a blindfold. And the "set" a
    set-level match reasons over is not complete until the receipt is in it.

    Every finding it raises belongs to an INVOICE, never to the note, so
    nothing is returned to the caller: they are written directly, and the ones
    that no longer hold are resolved. Never raises -- an unmatched receipt is
    still a captured receipt.
    """
    from src.services.extraction.three_way_match import check_against_receipts

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT po_id FROM proc.bp_goods_receipt_stg WHERE grn_id = %s",
                        (grn_id,))
            row = cur.fetchone()
            if row is None or not (row[0] or "").strip():
                return {"assessed": 0, "reason": "receipt_not_staged_or_no_po"}
            po_id = row[0]
            cur.execute(
                "SELECT line_no, item_description, quantity_received, "
                "       quantity_rejected, unit_of_measure, po_line_ref "
                "  FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id = %s "
                " ORDER BY line_no", (grn_id,))
            lines = [{"line_no": r[0], "item_description": r[1],
                      "quantity_received": r[2], "quantity_rejected": r[3],
                      "unit_of_measure": r[4], "po_line_ref": r[5]}
                     for r in cur.fetchall()]
        check_against_receipts("goods_receipt",
                               {"po_id": po_id, "grn_id": grn_id}, lines)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT lines_assessed, lines_unverifiable "
                        "FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (grn_id,))
            row = cur.fetchone()
        assessed, refused = (row or (0, 0))
        log.info("goods receipt %s matched: %s lines assessed, %s unverifiable",
                 grn_id, assessed, refused)
        return {"assessed": assessed or 0, "unverifiable": refused or 0,
                "reason": None}
    except Exception:  # noqa: BLE001 - a match must never lose a captured receipt
        log.exception("goods receipt %s: the three-way match failed", grn_id)
        return {"assessed": 0, "reason": "error"}
