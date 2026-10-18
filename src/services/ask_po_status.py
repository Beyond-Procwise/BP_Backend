"""PO-status questions: read every source that knows about the PO at once, join, and name what is missing (test F1/F3 shape).

"Where is PO-2025-0257?" is answered by several places that do not know about each other: the PO record, the invoices raised
against it, the goods receipts, and the approvals on its deal. Asked one after another, the slowest decides the wait and one
failure ends the answer. Here the sources are read in parallel threads, each with its own connection and its own time limit;
the answer waits for all of them (or their limits), and a source that failed or timed out is *named* as unavailable rather
than silently left out - so "no goods receipt recorded" and "goods receipts could not be read" never look alike.

There is no ERP integration in this codebase, so "the ERP" here is the procurement database of record. There is no carrier
tracking either, and none is built: shipment location is reported as not connected (test F3 needs it as a product decision).

This is in-process on purpose. ``orchestration.dag_scheduler`` runs named agent workflows through Redis workers and writes an
execution row per run, which is far more machinery (and a database write) than reading four tables for one question.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BRANCH_TIMEOUT_S = 5.0

_STATUS_WORDS = re.compile(
    r"\b(?:status|approved?|approval|open|closed|delivered|delivery|received|receipt|paid|payment|invoice[ds]?|billed|late|overdue|"
    r"progress|where\s+is|track(?:ing)?|outstanding|complete[d]?|pending|stage|state)\b", re.I)

NOT_CONNECTED = ("Carrier tracking", "not connected, so where a shipment is right now cannot be answered")

_SQL: Dict[str, Tuple[str, str]] = {
    "Purchase order record": (
        "SELECT po_id, supplier_name, po_status, approval_status, order_date, expected_delivery_date, currency, total_amount, "
        "total_amount_incl_tax, payment_terms, deal_name FROM proc.bp_purchase_order_trgt WHERE upper(po_id) = upper(%s)", "po"),
    "Invoices": (
        "SELECT invoice_id, invoice_date, due_date, invoice_paid_date, currency, invoice_total_incl_tax, invoice_status "
        "FROM proc.bp_invoice_trgt WHERE upper(po_id) = upper(%s) ORDER BY invoice_date LIMIT 10", "invoices"),
    "Goods receipts": (
        "SELECT grn_id, receipt_date, delivery_note_ref, received_by, lines_assessed, lines_unverifiable "
        "FROM proc.bp_goods_receipt_trgt WHERE upper(po_id) = upper(%s) ORDER BY receipt_date LIMIT 10", "receipts"),
    "Approvals": (
        "SELECT a.approval_id, a.status, a.decision, a.decision_reason, a.amount, a.currency, a.actioned_at "
        "FROM proc.bp_approval a JOIN proc.bp_purchase_order_trgt p ON p.deal_id = a.deal_id "
        "WHERE upper(p.po_id) = upper(%s) ORDER BY a.created_date DESC LIMIT 5", "approvals"),
}


def is_status_question(query: str, po_ids: List[str]) -> bool:
    """A question that names a PO and asks about its state (as opposed to asking what a PO is)."""
    return bool(po_ids) and bool(_STATUS_WORDS.search(query or ""))


def _read(conn_factory: Callable[[], Any], sql: str, po_id: str) -> List[Dict[str, Any]]:
    with conn_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (po_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def gather(conn_factory: Callable[[], Any], po_id: str, *, timeout_s: float = BRANCH_TIMEOUT_S,
           sources: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, Dict[str, Any]]:
    """{source name: {"state": "ok"|"empty"|"unavailable", "rows": [...], "why": str, "seconds": float}} for one PO.

    The branches run in parallel; the join waits for each up to its own limit. A branch past its limit is reported
    unavailable and its thread is left to finish on its own (a database call cannot be cancelled from here).
    """
    sources = sources or _SQL
    out: Dict[str, Dict[str, Any]] = {}
    started = {name: time.perf_counter() for name in sources}
    pool = cf.ThreadPoolExecutor(max_workers=len(sources), thread_name_prefix="po-status")
    futures = {name: pool.submit(_read, conn_factory, sql, po_id) for name, (sql, _) in sources.items()}
    try:
        for name, fut in futures.items():
            remaining = max(0.0, timeout_s - (time.perf_counter() - started[name]))
            try:
                rows = fut.result(timeout=remaining)
                out[name] = {"state": "ok" if rows else "empty", "rows": rows, "why": "", "seconds": round(time.perf_counter() - started[name], 3)}
            except cf.TimeoutError:
                out[name] = {"state": "unavailable", "rows": [], "why": f"did not answer within {timeout_s:g} seconds",
                             "seconds": round(time.perf_counter() - started[name], 3)}
            except Exception as exc:  # noqa: BLE001 - one source failing must not end the answer
                logger.warning("po-status source %s failed: %s", name, exc)
                out[name] = {"state": "unavailable", "rows": [], "why": "could not be read", "seconds": round(time.perf_counter() - started[name], 3)}
    finally:
        pool.shutdown(wait=False)
    return out


def _fmt(v: Any) -> str:
    return "" if v is None else str(v)


def render(po_id: str, results: Dict[str, Dict[str, Any]]) -> str:
    """The joined result as prompt text: what each source says, which sources are missing, and what is not connected."""
    blocks: List[str] = [f"Status of {po_id}, read from several sources at once:"]
    missing: List[str] = []
    for name, res in results.items():
        if res["state"] == "unavailable":
            missing.append(f"{name} ({res['why']})")
            continue
        if res["state"] == "empty":
            blocks.append(f"{name}: nothing recorded for this PO.")
            continue
        blocks.append(f"{name}:")
        for row in res["rows"]:
            blocks.append("- " + "; ".join(f"{k}={_fmt(v)}" for k, v in row.items() if v not in (None, "")))
    if missing:
        blocks.append("NOT AVAILABLE, so do not state or guess anything from these: " + "; ".join(missing) + ".")
    blocks.append(f"{NOT_CONNECTED[0]}: {NOT_CONNECTED[1]}.")
    return "\n".join(blocks)
