"""Cross-document reconciliation for a procurement deal.

Compares the final (_trgt) documents that share a ``deal_id`` and records the
matches/mismatches as consolidation actions in ``proc.bp_agent_actions``. This
is read-only on the _trgt data — it never mutates source records (extraction
accuracy mandate); the only writes are to the bp_agent_actions event log.

Checks (each independent and individually verifiable):
  - amount_usd : converted_amount_usd close across docs (currency-agnostic)
  - currency   : a single currency across the deal
  - supplier   : a single supplier across the deal
  - tax        : tax_percent close across docs
A dimension with fewer than two comparable values is ``skipped`` (logged
transparently rather than silently omitted).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from src.services.agent_actions import bulk_record, PHASE_CONSOLIDATION
from src.services.deal_summary import gather_deal_context

log = logging.getLogger(__name__)

# Tolerances (env-overridable).
_AMOUNT_PCT = float(os.getenv("RECON_AMOUNT_TOLERANCE_PCT", "0.01"))   # 1% of the largest
_AMOUNT_ABS = float(os.getenv("RECON_AMOUNT_TOLERANCE_ABS", "1.00"))   # or $1.00 floor
_TAX_PCT = float(os.getenv("RECON_TAX_TOLERANCE_PCT", "0.1"))          # 0.1 percentage points

# (doc_kind, ctx documents key, primary-key column)
_KINDS = [
    ("invoice", "invoices", "invoice_id"),
    ("purchase_order", "purchase_orders", "po_id"),
    ("quote", "quotes", "quote_id"),
]

# verdict status -> (action_type, agent_actions status)
_STATUS_TO_ACTION = {
    "match": ("reconcile_match", "ok"),
    "mismatch": ("reconcile_mismatch", "warn"),
    "skipped": ("reconcile_skipped", "skipped"),
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_docs(ctx: dict) -> list[dict]:
    """One row per document with the fields reconciliation compares."""
    docs: list[dict] = []
    for kind, ctx_key, pk_col in _KINDS:
        for r in ctx["documents"].get(ctx_key, []):
            docs.append({
                "doc_kind": kind,
                "doc_pk": r.get(pk_col),
                "amount_usd": _to_float(r.get("converted_amount_usd")),
                "currency": r.get("currency"),
                "supplier_id": r.get("supplier_id"),
                "tax_percent": _to_float(r.get("tax_percent")),
            })
    return docs


def _values_payload(docs: list[dict], field: str) -> list[dict]:
    return [
        {"doc_kind": d["doc_kind"], "doc_pk": d["doc_pk"], "value": d[field]}
        for d in docs
        if d[field] not in (None, "")
    ]


def _check_numeric(docs: list[dict], field: str, tol_fn: Callable[[list[float]], float],
                   dimension: str) -> dict:
    values = _values_payload(docs, field)
    nums = [v["value"] for v in values]
    if len(nums) < 2:
        return {"dimension": dimension, "status": "skipped", "values": values,
                "detail": "fewer than 2 comparable values"}
    spread = max(nums) - min(nums)
    tol = tol_fn(nums)
    status = "match" if spread <= tol else "mismatch"
    return {"dimension": dimension, "status": status, "values": values,
            "detail": f"spread={spread:.4f}, tolerance={tol:.4f}"}


def _check_distinct(docs: list[dict], field: str, dimension: str) -> dict:
    values = _values_payload(docs, field)
    if len(values) < 2:
        return {"dimension": dimension, "status": "skipped", "values": values,
                "detail": "fewer than 2 comparable values"}
    distinct = sorted({str(v["value"]) for v in values})
    status = "match" if len(distinct) <= 1 else "mismatch"
    return {"dimension": dimension, "status": status, "values": values,
            "detail": f"distinct={distinct}"}


def reconcile_deal(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Reconcile a deal's _trgt documents and write consolidation actions.

    Returns ``{deal_id, checks, actions_written}`` or ``None`` if the deal has no
    final records. The action rows are written via ``bulk_record`` — atomic with
    the caller's transaction when ``conn`` is provided, else autonomously.
    """
    ctx = gather_deal_context(deal_id, conn=conn)
    if ctx is None:
        return None

    docs = _flatten_docs(ctx)
    checks = {
        "amount_usd": _check_numeric(
            docs, "amount_usd",
            lambda nums: max(_AMOUNT_PCT * max(abs(n) for n in nums), _AMOUNT_ABS),
            "amount_usd",
        ),
        "currency": _check_distinct(docs, "currency", "currency"),
        "supplier": _check_distinct(docs, "supplier_id", "supplier"),
        "tax": _check_numeric(docs, "tax_percent", lambda nums: _TAX_PCT, "tax"),
    }

    action_rows = []
    for dimension, verdict in checks.items():
        action_type, status = _STATUS_TO_ACTION[verdict["status"]]
        action_rows.append({
            "phase": PHASE_CONSOLIDATION,
            "action_type": action_type,
            "deal_id": deal_id,
            "agent": "reconciler",
            "field_name": dimension,
            "status": status,
            "summary": f"{dimension} {verdict['status']} across {len(verdict['values'])} docs",
            "details": {
                "verdict": verdict["status"],
                "detail": verdict["detail"],
                "values": verdict["values"],
            },
        })
    bulk_record(action_rows, conn=conn)

    return {"deal_id": deal_id, "checks": checks, "actions_written": len(action_rows)}
