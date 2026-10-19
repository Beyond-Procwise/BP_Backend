"""Batch 2: the ``corpus_facts`` intents as named, row-scoped gateway functions.

Each function returns a dict keyed exactly as the legacy ``corpus_facts`` facts were, so the prompt text built from it is the
same. What changes is access: the legacy queries read the whole population for everyone; these read only the caller's buyers
(``buyer_id`` on invoices, quotes, POs and deals; findings through their document's deal). An empty scope runs no query.
Counted totals stay counted in SQL (never inferred from a top-N list), as the legacy comments explain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from src.services.version_collapse import latest_quote_pred

from .common import _P, _blind, _connect, _prov, _rows
from .context import CallerContext
from .envelope import Raw
from .gateway import Backends

_DOCTYPE = "CASE WHEN {c} IN ('po','purchase_order') THEN 'po' ELSE {c} END"


class OverviewParams(_P):
    limit: int = Field(default=10, ge=1, le=25)


def _where(ctx: CallerContext, column: Optional[str], *conds: str) -> Tuple[str, tuple]:
    """``WHERE ...`` for the fixed conditions plus the caller's buyer scope; no clause at all when there is nothing to say."""
    parts = [c for c in conds if c]
    params: tuple = ()
    if column and ctx.buyer_scope is not None:
        parts.append(f"{column}::text = ANY(%s)")
        params = (list(ctx.buyer_scope),)
    return ("WHERE " + " AND ".join(parts) if parts else ""), params


def _empty(*keys: str) -> Dict[str, List[Any]]:
    return {k: [] for k in keys}


def _out(data: Dict[str, Any], *tables: str) -> Raw:
    return Raw(data=data, provenance=[{**_prov(t, []), "aggregate": True} for t in tables])


def _q(cur: Any, sql: str, params: tuple = (), limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return _rows(cur, sql, params + ((limit,) if limit is not None else ()))


# --- suppliers ------------------------------------------------------------------------------------------------------------------

def get_supplier_overview(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    keys = ("totals", "suppliers_by_invoice_count", "suppliers_by_invoiced_amount")
    if _blind(ctx):
        return _out(_empty(*keys), "bp_invoice_trgt", "bp_supplier")
    where, sp = _where(ctx, "i.buyer_id")
    # The count of every supplier on record is the whole population; a restricted caller is not told it.
    on_record = ", (SELECT COUNT(*)::int FROM proc.bp_supplier) AS suppliers_on_record" if ctx.buyer_scope is None else ""
    with connect() as conn, conn.cursor() as cur:
        data = {
            "totals": _q(cur, f"SELECT COUNT(DISTINCT i.supplier_id)::int AS suppliers_we_have_invoices_from{on_record} "
                              f"FROM proc.bp_invoice_trgt i {where}", sp),
            "suppliers_by_invoice_count": _q(cur, "SELECT s.supplier_name, COUNT(i.invoice_id)::int AS invoices, "
                                                  "SUM(i.invoice_amount)::numeric AS invoiced_amount, MAX(i.currency) AS currency "
                                                  "FROM proc.bp_invoice_trgt i JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id "
                                                  f"{where} GROUP BY s.supplier_name "
                                                  "ORDER BY invoices DESC, invoiced_amount DESC NULLS LAST LIMIT %s", sp, p.limit),
            "suppliers_by_invoiced_amount": _q(cur, "SELECT s.supplier_name, SUM(i.invoice_amount)::numeric AS invoiced_amount, "
                                                    "MAX(i.currency) AS currency FROM proc.bp_invoice_trgt i "
                                                    "JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id "
                                                    f"{where} GROUP BY s.supplier_name ORDER BY invoiced_amount DESC NULLS LAST LIMIT %s",
                                               sp, p.limit),
        }
    return _out(data, "bp_invoice_trgt", "bp_supplier")


# --- invoices -------------------------------------------------------------------------------------------------------------------

def get_invoice_overview(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    keys = ("totals", "invoice_count_by_supplier", "largest_invoices_sample")
    if _blind(ctx):
        return _out(_empty(*keys), "bp_invoice_trgt")
    where, sp = _where(ctx, "i.buyer_id")
    with connect() as conn, conn.cursor() as cur:
        data = {
            "totals": _q(cur, "SELECT COUNT(*)::int AS invoices_total, COUNT(*) FILTER (WHERE i.po_id IS NOT NULL)::int AS citing_a_po "
                              f"FROM proc.bp_invoice_trgt i {where}", sp),
            "invoice_count_by_supplier": _q(cur, "SELECT s.supplier_name, COUNT(*)::int AS invoices FROM proc.bp_invoice_trgt i "
                                                 "JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id "
                                                 f"{where} GROUP BY s.supplier_name ORDER BY invoices DESC LIMIT %s", sp, p.limit),
            "largest_invoices_sample": _q(cur, "SELECT i.invoice_id, s.supplier_name, i.invoice_amount, i.currency, i.invoice_date, i.po_id "
                                               "FROM proc.bp_invoice_trgt i LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id "
                                               f"{where} ORDER BY i.invoice_amount DESC NULLS LAST LIMIT %s", sp, p.limit),
        }
    return _out(data, "bp_invoice_trgt", "bp_supplier")


# --- quotes ---------------------------------------------------------------------------------------------------------------------

def get_quote_overview(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    if _blind(ctx):
        return _out(_empty("totals", "quotes"), "bp_quote_trgt")
    where, sp = _where(ctx, "q.buyer_id")
    with connect() as conn, conn.cursor() as cur:
        data = {
            # quotes_total counts QUOTES (each bid once, at its latest version); quote_documents_total is the row count.
            "totals": _q(cur, f"SELECT COUNT(*) FILTER (WHERE {latest_quote_pred('q')})::int AS quotes_total, "
                              "COUNT(*)::int AS quote_documents_total, COUNT(DISTINCT q.supplier_id)::int AS suppliers_quoting "
                              f"FROM proc.bp_quote_trgt q {where}", sp),
            "quotes": _q(cur, "SELECT q.quote_id, s.supplier_name, q.total_amount, q.currency, q.quote_date FROM proc.bp_quote_trgt q "
                              "LEFT JOIN proc.bp_supplier s ON s.supplier_id = q.supplier_id "
                              f"{where} ORDER BY q.quote_date DESC NULLS LAST LIMIT %s", sp, p.limit),
        }
    return _out(data, "bp_quote_trgt", "bp_supplier")


# --- purchase orders ------------------------------------------------------------------------------------------------------------

def get_purchase_order_overview(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    if _blind(ctx):
        return _out(_empty("totals", "purchase_orders"), "bp_purchase_order_trgt")
    where, sp = _where(ctx, "p.buyer_id")
    with connect() as conn, conn.cursor() as cur:
        data = {
            "totals": _q(cur, "SELECT COUNT(*)::int AS purchase_orders_total, COUNT(DISTINCT p.supplier_id)::int AS suppliers_ordered_from "
                              f"FROM proc.bp_purchase_order_trgt p {where}", sp),
            "purchase_orders": _q(cur, "SELECT p.po_id, s.supplier_name, p.total_amount, p.currency, p.order_date "
                                       "FROM proc.bp_purchase_order_trgt p LEFT JOIN proc.bp_supplier s ON s.supplier_id = p.supplier_id "
                                       f"{where} ORDER BY p.total_amount DESC NULLS LAST LIMIT %s", sp, p.limit),
        }
    return _out(data, "bp_purchase_order_trgt", "bp_supplier")


# --- deals ----------------------------------------------------------------------------------------------------------------------

def list_deals(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    if _blind(ctx):
        return _out(_empty("deals", "documents_in_deals"), "bp_deal_overview", "bp_deal_documents")
    where_o, so = _where(ctx, "o.buyer_id")
    where_d, sd = _where(ctx, "d.buyer_id")
    with connect() as conn, conn.cursor() as cur:
        data = {
            # deal_date is empty on every row, so latest activity is what "recent" means.
            "deals": _q(cur, "SELECT o.deal_id, o.deal_name, o.supplier_name, o.deal_date, o.quote_count, to_jsonb(o)->>'bid_count' AS bid_count "
                             f"FROM proc.bp_deal_overview o {where_o} "
                             "ORDER BY COALESCE(o.deal_date, o.last_activity_date) DESC NULLS LAST, o.deal_id LIMIT %s", so, p.limit),
            "documents_in_deals": _q(cur, "SELECT d.deal_id, d.doc_type, d.doc_number, d.doc_date FROM proc.bp_deal_documents d "
                                          f"{where_d} ORDER BY d.deal_id, d.doc_date NULLS LAST LIMIT %s", sd, p.limit),
        }
    return _out(data, "bp_deal_overview", "bp_deal_documents")


# --- findings -------------------------------------------------------------------------------------------------------------------

def get_findings_overview(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    keys = ("totals", "open_findings_by_type", "largest_over_billing")
    if _blind(ctx):
        return _out(_empty(*keys), "bp_extraction_discrepancy")
    # A finding names its document; the document's deal names the buyer. EXISTS (not a join) so a document in two deals is
    # not counted twice. Findings with no deal are visible to an unrestricted caller only.
    scope, sp = "", ()
    if ctx.buyer_scope is not None:
        scope = ("AND EXISTS (SELECT 1 FROM proc.bp_deal_documents dd JOIN proc.bp_deal_overview o ON o.deal_id = dd.deal_id "
                 f"WHERE dd.doc_pk = d.doc_pk_candidate AND {_DOCTYPE.format(c='dd.doc_type')} = {_DOCTYPE.format(c='d.doc_type')} "
                 "AND o.buyer_id::text = ANY(%s))")
        sp = (list(ctx.buyer_scope),)
    with connect() as conn, conn.cursor() as cur:
        data = {
            # Totals first and stated outright: a model handed a truncated by-type list adds it up and calls the sum the total.
            "totals": _q(cur, "SELECT COUNT(*)::int AS open_findings_total, COUNT(*) FILTER (WHERE d.severity = 'critical')::int AS critical_total, "
                              "COUNT(DISTINCT d.issue_type)::int AS distinct_issue_types, COUNT(DISTINCT d.doc_pk_candidate)::int AS documents_affected "
                              f"FROM proc.bp_extraction_discrepancy d WHERE d.status = 'open' {scope}", sp),
            "open_findings_by_type": _q(cur, "SELECT d.issue_type, COUNT(*)::int AS findings, COUNT(*) FILTER (WHERE d.severity = 'critical')::int AS critical "
                                             f"FROM proc.bp_extraction_discrepancy d WHERE d.status = 'open' {scope} "
                                             "GROUP BY d.issue_type ORDER BY findings DESC LIMIT %s", sp, p.limit),
            "largest_over_billing": _q(cur, "SELECT d.doc_pk_candidate AS document, d.issue_type, d.severity, d.notes, "
                                            "NULLIF(d.computed_value, '')::numeric AS amount_over_po FROM proc.bp_extraction_discrepancy d "
                                            f"WHERE d.status = 'open' AND d.issue_type = 'amount_over_po' {scope} "
                                            "ORDER BY NULLIF(d.computed_value, '')::numeric DESC NULLS LAST LIMIT %s", sp, p.limit),
        }
    return _out(data, "bp_extraction_discrepancy", "bp_deal_documents")


# --- policies -------------------------------------------------------------------------------------------------------------------

def list_policies(ctx: CallerContext, p: OverviewParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    with connect() as conn, conn.cursor() as cur:
        rows = _q(cur, "SELECT policy_name, policy_type, policy_desc, version FROM proc.bp_policy "
                       "WHERE policy_status IS NULL OR policy_status = 1 ORDER BY policy_type, policy_name LIMIT %s", (), p.limit)
    return _out({"policies_in_force": rows}, "bp_policy")


DECLARED = {
    "get_supplier_overview": (OverviewParams, get_supplier_overview),
    "get_invoice_overview": (OverviewParams, get_invoice_overview),
    "get_quote_overview": (OverviewParams, get_quote_overview),
    "get_purchase_order_overview": (OverviewParams, get_purchase_order_overview),
    "list_deals": (OverviewParams, list_deals),
    "get_findings_overview": (OverviewParams, get_findings_overview),
    "list_policies": (OverviewParams, list_policies),
}
