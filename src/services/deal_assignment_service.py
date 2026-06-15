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

from src.services.db import get_conn
from src.services.linking_engine import (
    _PO,
    _norm_po,
    _rows,
    _table_columns,
    score_link,
)

log = logging.getLogger(__name__)

MIN_LINK_SCORE = float(os.getenv("PROMOTE_MIN_LINK_SCORE", "80"))
# Bar for a quote to ANCHOR a PO (form/complete a deal). Defaults to the link bar.
QUOTE_ANCHOR_MIN_SCORE = float(os.getenv("QUOTE_ANCHOR_MIN_SCORE",
                                         os.getenv("PROMOTE_MIN_LINK_SCORE", "80")))


# ---------------------------------------------------------------------------
# Pure helpers (no DB)
# ---------------------------------------------------------------------------
def basename_match(path_a: Optional[str], path_b: Optional[str]) -> bool:
    """True when two file paths share the same case-insensitive basename."""
    return _basename(path_a) != "" and _basename(path_a) == _basename(path_b)


def _basename(path: Optional[str]) -> str:
    """Normalized basename: directory-stripped, trimmed, lower-cased."""
    if not path:
        return ""
    return os.path.basename(str(path)).strip().lower()


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


# ---------------------------------------------------------------------------
# Document-table registry
# ---------------------------------------------------------------------------
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
_DOCTYPE_FROM_HINT = {"invoice": "invoice", "quote": "quote", "po": "po",
                      "purchase_order": "po", "purchaseorder": "po"}

# SQL fragment normalizing a po_id column the same way as _norm_po().
_PO_NORM_COND = "regexp_replace(regexp_replace(lower(po_id),'[^a-z0-9]','','g'),'^po','')"


def _persist_deal(cur, doc_type, doc_pk, *, deal_id, deal_name, document_id, deal_date):
    """Write deal columns onto the document's stg/trgt rows + their line items,
    only for columns that actually exist on each table."""
    pk, _raw, stg, trgt, line_stg, line_trgt = _DOC[doc_type]
    values = {"deal_id": deal_id, "deal_name": deal_name,
              "document_id": document_id, "deal_date": deal_date}
    for table in (stg, trgt, line_stg, line_trgt):
        cols = _table_columns(cur, table)
        present = [c for c in _DEAL_COLS if c in cols]
        if not present or pk not in cols:
            continue
        set_clause = ", ".join(f"{c}=%s" for c in present)
        cur.execute(
            f"update {table} set {set_clause} where {pk}=%s",
            [values[c] for c in present] + [doc_pk])


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


def _upsert_document_map(cur, deal_id, deal_name, doc_type, doc_pk, document_id, source_file):
    cur.execute(
        "insert into proc.bp_deal_document_map "
        "(document_id, deal_id, deal_name, doc_type, doc_pk, source_file) "
        "values (%s,%s,%s,%s,%s,%s) "
        "on conflict (document_id) do update set "
        "deal_id=excluded.deal_id, deal_name=excluded.deal_name, "
        "doc_type=excluded.doc_type, doc_pk=excluded.doc_pk, source_file=excluded.source_file",
        (document_id, deal_id, deal_name, doc_type, str(doc_pk), source_file))
    # A document belongs to exactly ONE current deal — drop any stale entries for
    # the same (doc_type, doc_pk) recorded under a previous deal_id/document_id.
    cur.execute(
        "delete from proc.bp_deal_document_map "
        "where doc_type=%s and doc_pk=%s and document_id<>%s",
        (doc_type, str(doc_pk), document_id))


def _prune_deal_document_map(cur) -> int:
    """Remove map rows whose document_id is no longer present in any _trgt table
    (the doc was re-keyed under a new deal, or removed). Keeps the map an exact
    mirror of the live deal->document relationships."""
    cur.execute(
        "delete from proc.bp_deal_document_map m where "
        "not exists (select 1 from proc.bp_invoice_trgt where document_id=m.document_id) and "
        "not exists (select 1 from proc.bp_quote_trgt where document_id=m.document_id) and "
        "not exists (select 1 from proc.bp_purchase_order_trgt where document_id=m.document_id)")
    return cur.rowcount or 0


def _propagate_deal_date(cur) -> int:
    """deal_date is a DEAL-level attribute (the order's expected delivery date).
    Compute it once per deal from the deal's PO(s) and stamp it on EVERY document
    in that deal, so a quote/invoice inherits the date even when it carries no PO
    reference of its own. Set-based; returns rows changed."""
    updated = 0
    deals = _rows(cur,
        "select deal_id, max(expected_delivery_date) dd from proc.bp_purchase_order_trgt "
        "where deal_id is not null and deal_id <> '' and expected_delivery_date is not null "
        "group by deal_id")
    for d in deals:
        for doc_type in ("invoice", "quote", "po"):
            _pk, _r, _s, trgt, _l, _lt = _DOC[doc_type]
            cur.execute(
                f"update {trgt} set deal_date=%s "
                f"where deal_id=%s and deal_date is distinct from %s",
                (d["dd"], d["deal_id"], d["dd"]))
            updated += cur.rowcount or 0
    return updated


def _raw_index(cur, raw_table, pk) -> dict:
    """One scan of a raw table -> {"by_pmid": {process_monitor_id: pk},
    "by_basename": {normalized basename(source_file): pk}}.

    Built once per raw table per run so the look-forward match is O(monitors),
    not O(monitors x raw-rows).
    """
    by_pmid: dict = {}     # process_monitor_id -> pk (exact)
    by_basename: dict = {}  # basename(source_file) -> pk, ONLY for pmid-less raws
    for r in _rows(cur, f"select {pk}, source_file, process_monitor_id from {raw_table}") or []:
        pmid = r.get("process_monitor_id")
        if pmid is not None:
            # A raw that knows its triggering monitor row is matched ONLY by that
            # exact id — never by basename. Otherwise other monitor rows sharing
            # the filename (same file re-uploaded under a different deal) would
            # cross-claim it, nondeterministically overwriting the right deal.
            by_pmid[pmid] = r.get(pk)
            continue
        key = _basename(r.get("source_file"))
        if key:
            by_basename[key] = r.get(pk)
    return {"by_pmid": by_pmid, "by_basename": by_basename}


def _ensure_in_trgt(cur, doc_type, doc_pk) -> bool:
    """Copy a staged doc (+ its line items) into _trgt if it isn't there yet.

    This is the deal-path promotion: a deal-tagged document reaches _trgt
    independent of PO linkage (the PO-gated linking_engine.promote_ready holds
    PO-less docs forever). Reuses the linking_engine copy helpers, which exclude
    the deal columns — _look_forward stamps those afterward. Returns True when
    the doc is present in _trgt afterward.
    """
    from src.services.linking_engine import _copyable_cols, _upsert, _copy_lines
    pk, _raw, stg, trgt, line_stg, line_trgt = _DOC[doc_type]
    if _rows(cur, f"select 1 from {trgt} where {pk}=%s limit 1", (doc_pk,)):
        return True
    staged = _rows(cur, f"select * from {stg} where {pk}=%s", (doc_pk,))
    if not staged:
        return False   # not promoted to _stg yet (e.g. held at raw by a discrepancy)
    cols = _copyable_cols(cur, stg, trgt)
    if not cols:
        return False
    _upsert(cur, trgt, pk, staged[0], cols)
    _copy_lines(cur, pk, doc_pk, line_stg, line_trgt)
    return True


def _look_forward(cur) -> int:
    """Stamp process_monitor deals onto matching extracted documents.

    A document is matched to its monitor row by process_monitor_id (exact, set by
    the RENOVATION extraction path) when available, else by basename(file_path) ==
    basename(source_file). IDEMPOTENT: a doc already carrying its exact deal_id is
    skipped (not re-stamped, not counted), so a stable system returns 0 and does
    not retrigger downstream mining. Status is owned solely by reconcile_status.
    Returns the number of documents NEWLY linked this run."""
    linked = 0
    # Ordered by id so that, when the SAME document is claimed by several monitor
    # uploads tagged with DIFFERENT deals (a conflict), the FIRST (lowest-id)
    # upload deterministically owns it this run — making the pass stable/idempotent
    # rather than flapping every cycle. _flag_conflict_po_chains still surfaces the
    # disagreement for resolution.
    monitors = _rows(cur,
        "select id, file_path, deal_id, deal_name, category, document_type "
        "from proc.process_monitor "
        "where deal_id is not null and deal_id <> '' order by id")
    raw_index_cache: dict = {}   # raw_table -> {"by_pmid": {...}, "by_basename": {...}}
    claimed: set = set()         # (doc_type, doc_pk) already owned this run
    for m in monitors:
        m_key = _basename(m.get("file_path"))
        for dt in _candidate_doc_types(m.get("category"), m.get("document_type")):
            pk, raw, _stg, trgt, _ls, _lt = _DOC[dt]
            if raw not in raw_index_cache:
                raw_index_cache[raw] = _raw_index(cur, raw, pk)
            idx = raw_index_cache[raw]
            doc_pk = idx["by_pmid"].get(m["id"])
            if not doc_pk and m_key:
                doc_pk = idx["by_basename"].get(m_key)
            if not doc_pk:
                continue
            if (dt, doc_pk) in claimed:
                continue   # a lower-id upload already owns this doc this run
            claimed.add((dt, doc_pk))
            # Already linked to this exact deal? -> nothing to do (idempotent).
            cur_d = _rows(cur, f"select deal_id from {trgt} where {pk}=%s", (doc_pk,))
            if cur_d and (cur_d[0].get("deal_id") or "") == (m["deal_id"] or ""):
                continue
            # Deal-path promotion: a deal-tagged doc must reach _trgt even when it
            # has no parent PO (the PO-gated promote_ready holds those). Copy the
            # staged row into _trgt first; the deal grouping is authoritative.
            _ensure_in_trgt(cur, dt, doc_pk)
            doc_id = mint_document_id(m["deal_id"], dt, str(doc_pk))
            deal_date = _deal_date_for_doc(cur, dt, doc_pk)
            _persist_deal(cur, dt, doc_pk, deal_id=m["deal_id"], deal_name=m["deal_name"],
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, m["deal_id"], m["deal_name"], dt, doc_pk, doc_id,
                                 m["file_path"])
            linked += 1
    return linked


def _invoice_line_delivery(cur, doc_pk):
    """Best-effort invoice-line delivery_date fallback for deal_date."""
    rows = _rows(cur,
        f"select delivery_date from {_DOC['invoice'][5]} where invoice_id=%s "
        f"and delivery_date is not null limit 1", (doc_pk,))
    return rows[0].get("delivery_date") if rows else None


def _deal_date_for_doc(cur, doc_type, doc_pk):
    """Resolve the order's expected delivery date for the deal this doc belongs to."""
    # po doc: its own expected_delivery_date
    if doc_type == "po":
        r = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where po_id=%s", (doc_pk,))
        return resolve_deal_date(r[0] if r else None)
    # invoice/quote: find the parent PO via po_id on the doc, then its delivery date
    pk, _raw, stg, trgt, _ls, _lt = _DOC[doc_type]
    rr = _rows(cur, f"select po_id from {trgt} where {pk}=%s", (doc_pk,)) or \
         _rows(cur, f"select po_id from {stg} where {pk}=%s", (doc_pk,))
    po_ref = rr[0].get("po_id") if rr else None
    inv_line = _invoice_line_delivery(cur, doc_pk) if doc_type == "invoice" else None
    if not po_ref:
        return resolve_deal_date(None, inv_line)
    pr = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where {_PO_NORM_COND}=%s",
               (_norm_po(po_ref),))
    return resolve_deal_date(pr[0] if pr else None, inv_line)


# ---------------------------------------------------------------------------
# Look-back pass (quote-gated)
# ---------------------------------------------------------------------------
def _quote_anchor_for_po(cur, po, npo=None):
    """Return the quote that ANCHORS this PO (or None). Quote is the deal anchor:
    matched by explicit reference (quote.po_id, po_line_items.quote_number) or, when
    refs are absent, by relationship score >= QUOTE_ANCHOR_MIN_SCORE (supplier +
    line-item/product overlap + amount + temporal). Candidates narrowed by supplier."""
    quo = _DOC["quote"][3]
    npo = npo or _norm_po(po.get("po_id"))
    # 1) explicit: a quote whose own po_id resolves to this canonical PO
    if npo:
        ex = _rows(cur, f"select * from {quo} where {_PO_NORM_COND}=%s", (npo,))
        if ex:
            return ex[0]
    # 2) the PO's line items naming a quote_number
    poln = _DOC["po"][5]
    for r in _rows(cur, f"select distinct quote_number from {poln} "
                        f"where po_id=%s and coalesce(quote_number,'')<>''", (po.get("po_id"),)):
        qr = _rows(cur, f"select * from {quo} where quote_id=%s", (str(r["quote_number"]),))
        if qr:
            return qr[0]
    # 3) relationship score (supplier-narrowed; line items strengthen the match)
    sup = po.get("supplier_id")
    cand = (_rows(cur, f"select * from {quo} where supplier_id=%s", (sup,)) if sup
            else _rows(cur, f"select * from {quo}"))
    po_lines = _rows(cur, f"select * from {poln} where po_id=%s", (po.get("po_id"),))
    best, best_f = None, 0.0
    for q in cand:
        q_lines = _rows(cur, f"select * from {_DOC['quote'][5]} where quote_id=%s", (q.get("quote_id"),))
        link = score_link(q, po, "quote_po", source_lines=q_lines, target_lines=po_lines)
        if link.get("F", 0) >= QUOTE_ANCHOR_MIN_SCORE and link["F"] > best_f:
            best, best_f = q, link["F"]
    return best


def _look_back(cur) -> int:
    """Quote-gated deal formation. A deal forms for a canonical PO ONLY when a quote
    anchors it (Quote -> PO -> Invoice flow). The anchoring quote, the PO, and the
    PO's invoices are assigned to the deal (the PO's existing authoritative deal, or a
    derived DEALV2-<po>). POs/invoices with no anchoring quote are left orphaned (no
    deal minted) — surfaced as Orphaned_Awaiting_Quote by reconcile_status. Only
    currently-unlinked docs are stamped, so authoritative look-forward deals are never
    clobbered. Returns the number of documents newly linked."""
    inv_trgt = _DOC["invoice"][3]
    linked = 0
    for po in _rows(cur, f"select * from {_PO['trgt']}"):
        npo = _norm_po(po.get("po_id"))
        if not npo:
            continue
        quote = _quote_anchor_for_po(cur, po, npo)
        if quote is None:
            continue   # no anchoring quote -> PO + its invoices stay orphaned
        supplier = po.get("supplier_name") or po.get("supplier_id")
        existing = (po.get("deal_id") or "").strip() or (quote.get("deal_id") or "").strip()
        deal_id = existing or lookback_deal_id(npo)
        deal_name = ((po.get("deal_name") or quote.get("deal_name")) if existing
                     else lookback_deal_name(supplier, npo))
        deal_date = resolve_deal_date(po)
        members = [("po", po.get("po_id"), po.get("deal_id")),
                   ("quote", quote.get("quote_id"), quote.get("deal_id"))]
        for inv in _rows(cur, f"select invoice_id, deal_id from {inv_trgt} where {_PO_NORM_COND}=%s", (npo,)):
            members.append(("invoice", inv["invoice_id"], inv.get("deal_id")))
        for dt, dpk, cur_deal in members:
            if (cur_deal or "").strip():
                continue   # already linked -> never clobber an authoritative deal
            doc_id = mint_document_id(deal_id, dt, str(dpk))
            _persist_deal(cur, dt, dpk, deal_id=deal_id, deal_name=deal_name,
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, dt, dpk, doc_id, None)
            linked += 1
    return linked


def _po_chain_groups(cur) -> dict:
    """canonical_po -> list of (doc_type, doc_pk, deal_id, deal_name) for every
    Quote/PO/Invoice referencing that PO. Shared by propagation and conflict
    detection so both reason over the same grouping."""
    groups: dict = {}
    for doc_type in ("invoice", "quote", "po"):
        pk, _raw, _stg, trgt, _ls, _lt = _DOC[doc_type]
        if doc_type == "po":
            rows = _rows(cur, f"select po_id, deal_id, deal_name from {trgt}")
            for r in rows:
                npo = _norm_po(r.get("po_id"))
                if npo:
                    groups.setdefault(npo, []).append(("po", r["po_id"], r.get("deal_id"), r.get("deal_name")))
        else:
            rows = _rows(cur, f"select {pk}, po_id, deal_id, deal_name from {trgt}")
            for r in rows:
                npo = _norm_po(r.get("po_id"))
                if npo:
                    groups.setdefault(npo, []).append((doc_type, r[pk], r.get("deal_id"), r.get("deal_name")))
    return groups


def _po_chain_deal_ids(members) -> set:
    """Distinct non-blank deal_ids among a PO chain's documents."""
    return {d for (_t, _pk, d, _n) in members if d}


def _propagate_deal_along_po(cur) -> int:
    """Spread a known deal across the full PO chain.

    Every Quote/PO/Invoice that shares a canonical PO belongs to one deal. When
    any doc on a PO already carries a deal_id (from look-forward tagging or
    look-back), stamp that same deal onto the PO-chain siblings that have none —
    so the complete Quote->PO->Invoice chain lands in a single deal even if only
    one document was tagged. POs whose docs disagree on the deal are left
    untouched (flagged by _flag_conflict_po_chains). Idempotent.
    """
    updated = 0
    for members in _po_chain_groups(cur).values():
        deal_ids = _po_chain_deal_ids(members)
        if len(deal_ids) != 1:
            continue   # 0 deals -> nothing to spread; >1 -> conflict (flagged elsewhere)
        deal_id = next(iter(deal_ids))
        deal_name = next((n for (_t, _pk, d, n) in members if d == deal_id), None)
        for (dt, dpk, existing, _n) in members:
            if existing:
                continue
            document_id = mint_document_id(deal_id, dt, str(dpk))
            deal_date = _deal_date_for_doc(cur, dt, dpk)
            _persist_deal(cur, dt, dpk, deal_id=deal_id, deal_name=deal_name,
                          document_id=document_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, dt, dpk, document_id, None)
            updated += 1
    return updated


def _monitor_ids_for_doc(cur, doc_type, doc_pk) -> list:
    """Every process_monitor_id that produced this document (a re-uploaded file
    yields several raw rows / monitor rows for the same doc_pk)."""
    pk, raw, _stg, _trgt, _ls, _lt = _DOC[doc_type]
    return [r["process_monitor_id"] for r in _rows(
        cur, f"select process_monitor_id from {raw} "
             f"where {pk}=%s and process_monitor_id is not null", (doc_pk,))]


def _flag_conflict_po_chains(cur) -> int:
    """Flag PO chains whose documents carry MORE THAN ONE distinct deal_id.

    Propagation deliberately leaves such chains untouched (it can't know which
    deal is right). Here we surface the mismatch for human resolution by setting
    process_monitor.status='Deal_Conflict_Review' on every monitor row behind a
    deal-bearing doc in the conflicted chain. Returns the count of monitor rows
    flagged. Idempotent (re-setting the same status is a no-op in effect).
    """
    flagged: set = set()
    for members in _po_chain_groups(cur).values():
        if len(_po_chain_deal_ids(members)) <= 1:
            continue
        for (dt, dpk, deal_id, _n) in members:
            if not deal_id:
                continue
            for mid in _monitor_ids_for_doc(cur, dt, dpk):
                if mid not in flagged:
                    _set_monitor_status(cur, mid, "Deal_Conflict_Review")
                    flagged.add(mid)
    return len(flagged)


# ---------------------------------------------------------------------------
# Reconciliation + orchestration
# ---------------------------------------------------------------------------
def _reconcile_legacy(cur) -> int:
    """Rewrite legacy DEAL-<po> keys (no monitor deal) to the derived DEALV2-<po>
    form so all deal_ids share one scheme. Authoritative monitor deals already
    overwrote their rows in the look-forward pass."""
    reconciled = 0
    for doc_type in ("invoice", "quote", "po"):
        pk, _raw, _stg, trgt, _ls, _lt = _DOC[doc_type]
        # Pass the LIKE pattern as a bind param — a bare '%' in the SQL string
        # would be misread by psycopg2 as a parameter placeholder.
        rows = _rows(cur,
            f"select {pk}, deal_id, po_id from {trgt} where deal_id like %s",
            ("DEAL-%",))
        for r in rows:
            npo = _norm_po(r.get("po_id")) or r["deal_id"].split("-", 1)[-1]
            new_id = lookback_deal_id(npo)
            if new_id == r["deal_id"]:
                continue
            cur.execute(f"update {trgt} set deal_id=%s where {pk}=%s", (new_id, r[pk]))
            reconciled += 1
    return reconciled


def _backfill_deal_metadata(cur) -> int:
    """Stamp document_id (and deal_date where resolvable) on every _trgt doc that
    has a deal_id but no document_id yet — e.g. legacy rows that were only
    reconciled, or rows assigned by an external process. Idempotent: once
    document_id is set, the row is skipped on the next run."""
    updated = 0
    for doc_type in ("invoice", "quote", "po"):
        pk, _raw, _stg, trgt, _ls, _lt = _DOC[doc_type]
        rows = _rows(cur,
            f"select {pk}, deal_id, deal_name from {trgt} "
            f"where deal_id is not null and deal_id <> '' "
            f"and (document_id is null or document_id = '')")
        for r in rows:
            doc_pk = r[pk]
            document_id = mint_document_id(r["deal_id"], doc_type, str(doc_pk))
            deal_date = _deal_date_for_doc(cur, doc_type, doc_pk)
            _persist_deal(cur, doc_type, doc_pk, deal_id=r["deal_id"],
                          deal_name=r.get("deal_name"), document_id=document_id,
                          deal_date=deal_date)
            _upsert_document_map(cur, r["deal_id"], r.get("deal_name"), doc_type,
                                 doc_pk, document_id, None)
            updated += 1
    return updated


def reconcile_status(cur) -> int:
    """Set each document's process_monitor.status to reflect its TRUE furthest
    pipeline stage AND the quote-anchored deal model:

        Deal_Linked              — quote in _trgt with a deal; OR a PO/invoice whose
                                   deal HAS a quote (a complete Quote->PO->Invoice chain)
        Orphaned_Awaiting_Quote  — a PO/invoice in _trgt whose deal has NO quote
                                   (or has no deal) — the quote anchor is missing
        Deal_Unassigned_Review   — a quote in _trgt with no deal
        Staged                   — promoted to _stg, not yet in _trgt (gate-held)
        Discrepancy_Review       — held at _raw by a blocking discrepancy
        Extracted                — in _raw, clean, not yet staged

    Quotes are never orphaned (the quote IS the anchor). Set-based, one UPDATE per
    doc type. Never touches Extraction_Failed or Deal_Conflict_Review. Authoritative
    for the deal-stage statuses (quote presence, not deal_id alone, decides
    Deal_Linked vs Orphaned), so it does NOT skip deal-tagged rows. Returns rows changed.
    """
    # PO/invoice: complete only when their deal contains a quote; else orphaned.
    case_po_inv = (
        "case "
        "when t.{pk} is not null and t.deal_id is not null and t.deal_id <> '' "
        "  and exists(select 1 from proc.bp_quote_trgt q where q.deal_id = t.deal_id) then 'Deal_Linked' "
        "when t.{pk} is not null then 'Orphaned_Awaiting_Quote' "
        "when s.{pk} is not null then 'Staged' "
        "when r.promotion_status = 'discrepancy' then 'Discrepancy_Review' "
        "else 'Extracted' end")
    # Quote: anchor — linked when it has a deal, else unassigned (never orphaned).
    case_quote = (
        "case "
        "when t.{pk} is not null and t.deal_id is not null and t.deal_id <> '' then 'Deal_Linked' "
        "when t.{pk} is not null then 'Deal_Unassigned_Review' "
        "when s.{pk} is not null then 'Staged' "
        "when r.promotion_status = 'discrepancy' then 'Discrepancy_Review' "
        "else 'Extracted' end")
    updated = 0
    for doc_type in ("invoice", "quote", "po"):
        pk, raw, stg, trgt, _ls, _lt = _DOC[doc_type]
        st_case = (case_quote if doc_type == "quote" else case_po_inv).format(pk=pk)
        cur.execute(
            f"""
            update proc.process_monitor pm
               set status = sub.st, lastmodified_date = now()
            from (
              select r.process_monitor_id pm_id, {st_case} st
              from {raw} r
              left join {stg} s  on s.{pk} = r.{pk}
              left join {trgt} t on t.{pk} = r.{pk}
              where r.process_monitor_id is not null
            ) sub
            where pm.id = sub.pm_id
              and pm.status not in ('Extraction_Failed', 'Deal_Conflict_Review')
              and pm.status is distinct from sub.st
            """)
        updated += cur.rowcount or 0
    return updated


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
    prop = _propagate_deal_along_po(cur)
    conflicts = _flag_conflict_po_chains(cur)
    meta = _backfill_deal_metadata(cur)
    dates = _propagate_deal_date(cur)
    pruned = _prune_deal_document_map(cur)
    status = reconcile_status(cur)
    return {"forward_linked": fwd, "backward_linked": back, "reconciled": rec,
            "propagated": prop, "conflicts_flagged": conflicts,
            "metadata_filled": meta, "deal_dates_set": dates,
            "map_pruned": pruned, "status_reconciled": status}
