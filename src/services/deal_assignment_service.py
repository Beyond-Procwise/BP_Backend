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


from src.services.linking_engine import _rows

_DOCTYPE_FROM_HINT = {"invoice": "invoice", "quote": "quote", "po": "po",
                      "purchase_order": "po", "purchaseorder": "po"}


def _set_monitor_status(cur, monitor_id, status):
    cur.execute(
        "update proc.process_monitor set status=%s, lastmodified_date=now() where id=%s",
        (status, monitor_id))


def _candidate_doc_types(category, document_type):
    for hint in (category, document_type):
        key = (hint or "").strip().lower().replace(" ", "")
        if key in _DOCTYPE_FROM_HINT:
            return [_DOCTYPE_FROM_HINT[key]]
    return ["invoice", "quote", "po"]   # unknown hint -> search all


def _look_forward(cur) -> int:
    """Stamp process_monitor deals onto matching extracted documents."""
    linked = 0
    monitors = _rows(cur,
        "select id, file_path, deal_id, deal_name, category, document_type "
        "from proc.process_monitor "
        "where deal_id is not null and deal_id <> ''")
    for m in monitors:
        matched = False
        for dt in _candidate_doc_types(m.get("category"), m.get("document_type")):
            pk, raw, stg, trgt, _ls, _lt = _DOC[dt]
            # find the doc whose source_file basename matches the monitor file_path
            src_rows = _rows(cur, f"select {pk}, source_file from {raw}") or []
            hit = next((r for r in src_rows
                        if basename_match(m["file_path"], r.get("source_file"))), None)
            if not hit:
                # raw may be absent; fall back to trgt by basename of any source col if present
                continue
            doc_pk = hit[pk]
            doc_id = mint_document_id(m["deal_id"], dt, str(doc_pk))
            # resolve deal_date from the deal's PO if this is/has one (best-effort)
            deal_date = _deal_date_for_doc(cur, dt, doc_pk)
            _persist_deal(cur, dt, doc_pk, deal_id=m["deal_id"], deal_name=m["deal_name"],
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, m["deal_id"], m["deal_name"], dt, doc_pk, doc_id,
                                 m["file_path"])
            matched = True
            linked += 1
        _set_monitor_status(cur, m["id"], "Deal_Linked" if matched
                            else "Deal_Unassigned_Review")
    return linked


def _upsert_document_map(cur, deal_id, deal_name, doc_type, doc_pk, document_id, source_file):
    cur.execute(
        "insert into proc.bp_deal_document_map "
        "(document_id, deal_id, deal_name, doc_type, doc_pk, source_file) "
        "values (%s,%s,%s,%s,%s,%s) "
        "on conflict (document_id) do update set "
        "deal_id=excluded.deal_id, deal_name=excluded.deal_name, "
        "doc_type=excluded.doc_type, doc_pk=excluded.doc_pk, source_file=excluded.source_file",
        (document_id, deal_id, deal_name, doc_type, str(doc_pk), source_file))


def _deal_date_for_doc(cur, doc_type, doc_pk):
    """Resolve the order's expected delivery date for the deal this doc belongs to."""
    from src.services.linking_engine import _PO, _norm_po
    # po doc: its own expected_delivery_date
    if doc_type == "po":
        r = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where po_id=%s", (doc_pk,))
        return r[0]["expected_delivery_date"] if r else None
    # invoice/quote: find parent PO via po_id on the doc, then its delivery date
    pk, _raw, stg, trgt, _ls, _lt = _DOC[doc_type]
    rr = _rows(cur, f"select po_id from {trgt} where {pk}=%s", (doc_pk,)) or \
         _rows(cur, f"select po_id from {stg} where {pk}=%s", (doc_pk,))
    po_ref = rr[0].get("po_id") if rr else None
    if not po_ref:
        return None
    cond = "regexp_replace(regexp_replace(lower(po_id),'[^a-z0-9]','','g'),'^po','')"
    pr = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where {cond}=%s",
               (_norm_po(po_ref),))
    return pr[0]["expected_delivery_date"] if pr else None


from src.services.linking_engine import score_link, _norm_po, _PO

MIN_LINK_SCORE = float(os.getenv("PROMOTE_MIN_LINK_SCORE", "80"))


def _unlinked_docs(cur, doc_type):
    pk, _raw, stg, trgt, _ls, _lt = _DOC[doc_type]
    return _rows(cur, f"select * from {trgt} where deal_id is null or deal_id = ''")


def _look_back(cur) -> int:
    """Group deal-less docs under their canonical PO deal when the link score passes."""
    linked = 0
    for doc_type in ("invoice", "quote"):
        pk = _DOC[doc_type][0]
        for row in _unlinked_docs(cur, doc_type):
            po_ref = row.get("po_id")
            npo = _norm_po(po_ref)
            if not npo:
                continue   # no-PO doc -> left for review by orchestrator
            cond = "regexp_replace(regexp_replace(lower(po_id),'[^a-z0-9]','','g'),'^po','')"
            pos = _rows(cur, f"select * from {_PO['trgt']} where {cond}=%s", (npo,))
            if not pos:
                continue
            po = pos[0]
            profile = "invoice_po" if doc_type == "invoice" else "quote_po"
            link = score_link(row, po, profile)
            if link["F"] < MIN_LINK_SCORE:
                continue
            supplier = po.get("supplier_name") or po.get("supplier_id")
            deal_id = lookback_deal_id(npo)
            deal_name = lookback_deal_name(supplier, npo)
            doc_pk = row[pk]
            doc_id = mint_document_id(deal_id, doc_type, str(doc_pk))
            deal_date = po.get("expected_delivery_date")
            _persist_deal(cur, doc_type, doc_pk, deal_id=deal_id, deal_name=deal_name,
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, doc_type, doc_pk, doc_id,
                                 row.get("source_file"))
            # also stamp the PO itself into the same derived deal
            po_doc_id = mint_document_id(deal_id, "po", str(po["po_id"]))
            _persist_deal(cur, "po", po["po_id"], deal_id=deal_id, deal_name=deal_name,
                          document_id=po_doc_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, "po", po["po_id"], po_doc_id, None)
            linked += 1
    return linked


from src.services.db import get_conn


def _reconcile_legacy(cur) -> int:
    """Rewrite legacy DEAL-<po> keys (no monitor deal) to the derived DEALV2-<po>
    form so all deal_ids share one scheme. Authoritative monitor deals already
    overwrote their rows in the look-forward pass."""
    reconciled = 0
    for doc_type in ("invoice", "quote", "po"):
        pk, _raw, _stg, trgt, _ls, _lt = _DOC[doc_type]
        rows = _rows(cur,
            f"select {pk}, deal_id, po_id from {trgt} "
            f"where deal_id like 'DEAL-%'")
        for r in rows:
            npo = _norm_po(r.get("po_id")) or r["deal_id"].split("-", 1)[-1]
            new_id = lookback_deal_id(npo)
            if new_id == r["deal_id"]:
                continue
            cur.execute(f"update {trgt} set deal_id=%s where {pk}=%s", (new_id, r[pk]))
            reconciled += 1
    return reconciled


def _flag_unassigned(cur) -> int:
    """Mark monitor rows whose document still has no deal as review-needed."""
    cur.execute(
        "update proc.process_monitor set status='Deal_Unassigned_Review', "
        "lastmodified_date=now() "
        "where (deal_id is null or deal_id='') "
        "and status not in ('Extraction_Failed','Deal_Unassigned_Review') "
        "returning id")
    try:
        return len(cur.fetchall())
    except Exception:
        return 0


def assign_deals(conn: Any = None, limit: Optional[int] = None) -> dict:
    """Run look-forward, look-back, reconcile, and unassigned-flag passes."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                r = _run(own.cursor())
                own.commit()
                return r
            except Exception:
                own.rollback()
                raise
    return _run(conn.cursor())


def _run(cur) -> dict:
    fwd = _look_forward(cur)
    back = _look_back(cur)
    rec = _reconcile_legacy(cur)
    flag = _flag_unassigned(cur)
    return {"forward_linked": fwd, "backward_linked": back,
            "reconciled": rec, "unassigned_review": flag}
