"""Assign procurement documents to deals.

deal_id is the backend grouping key. For look-forward documents it is taken
verbatim from proc.process_monitor (user-supplied at upload). For look-back
documents it is derived deterministically from the canonical PO, using the
existing linking_engine score to decide membership. deal_name is the
user-facing label; document_id is a stable per-document id within a deal;
deal_date is the order's expected delivery date stamped on every doc.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)


def basename_match(path_a: Optional[str], path_b: Optional[str]) -> bool:
    """True when two file paths share the same case-insensitive basename."""
    if not path_a or not path_b:
        return False
    ba = os.path.basename(str(path_a)).strip().lower()
    bb = os.path.basename(str(path_b)).strip().lower()
    return bool(ba) and ba == bb


def mint_document_id(deal_id: str, doc_type: str, doc_pk: str) -> str:
    """Deterministic per-document identity within a deal."""
    return f"{deal_id}::{doc_type}::{doc_pk}"


def resolve_deal_date(po_row: Optional[dict], inv_line_delivery=None):
    """deal_date = order expected delivery date.

    Prefer the deal's PO expected_delivery_date; fall back to an invoice line
    delivery_date; else None.
    """
    if po_row and po_row.get("expected_delivery_date"):
        return po_row["expected_delivery_date"]
    return inv_line_delivery


def lookback_deal_id(canonical_po: str) -> str:
    """Versioned derived deal_id (avoids colliding with legacy DEAL-<po>)."""
    return f"DEALV2-{canonical_po}"


def lookback_deal_name(supplier_name: Optional[str], canonical_po: str) -> str:
    supplier = (supplier_name or "Unknown Supplier").strip()
    return f"{supplier} — PO {canonical_po}"


from src.services.linking_engine import _table_columns  # column introspection

# doc_type -> (pk, raw, stg, trgt, line_stg, line_trgt)
_DOC = {
    "invoice": ("invoice_id",
                "proc.bp_invoice_raw", "proc.bp_invoice_stg", "proc.bp_invoice_trgt",
                "proc.bp_invoice_line_items_stg", "proc.bp_invoice_line_items_trgt"),
    "quote": ("quote_id",
              "proc.bp_quote_raw", "proc.bp_quote_stg", "proc.bp_quote_trgt",
              "proc.bp_quote_line_items_stg", "proc.bp_quote_line_items_trgt"),
    "po": ("po_id",
           "proc.bp_purchase_order_raw", "proc.bp_purchase_order_stg", "proc.bp_purchase_order_trgt",
           "proc.bp_po_line_items_stg", "proc.bp_po_line_items_trgt"),
}
_DEAL_COLS = ("deal_id", "deal_name", "document_id", "deal_date")


def _persist_deal(cur, doc_type, doc_pk, *, deal_id, deal_name, document_id, deal_date):
    """Write deal columns onto the document's stg/trgt rows + their line items,
    only for columns that actually exist on each table."""
    pk, _raw, stg, trgt, line_stg, line_trgt = _DOC[doc_type]
    values = {"deal_id": deal_id, "deal_name": deal_name,
              "document_id": document_id, "deal_date": deal_date}
    for table in (stg, trgt, line_stg, line_trgt):
        present = [c for c in _DEAL_COLS if c in _table_columns(cur, table)]
        if not present or pk not in _table_columns(cur, table):
            continue
        set_clause = ", ".join(f"{c}=%s" for c in present)
        cur.execute(
            f"update {table} set {set_clause} where {pk}=%s",
            [values[c] for c in present] + [doc_pk])
