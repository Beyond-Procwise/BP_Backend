"""Verify L3 judge wiring by re-running 3 stuck docs through dispatch.

Pick docs that previously hit `discrepancy` with specific missing_required
fields, dispatch them again, then compare what the renovation pipeline now
extracts. We use the Ollama backend for the judge (text-only, fastest
verification path) — Qwen2.5-VL is the prod default but we want this
script to be cheap enough to run on demand.

This script is read-only on the source files (S3 keys). It DOES write new
rows to proc.bp_<doctype>_raw via dispatch_document — that's the point:
new rows let us compare before/after on a stable doc.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force the lightweight grounded-judge backend for verification.
os.environ.setdefault("EXTRACTION_V3_JUDGE_MODEL", "ollama")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.dispatch import dispatch_document  # noqa: E402


# Three docs that all failed in the most recent batch — one per doc_type.
CASES = [
    {
        "doc_type": "purchase_order",
        "file_path": "documents/po/DESIGN HOUSE AGENCY PO438295 for QUT DHA-2025-102 .pdf",
        "previously_missing": ["supplier_name"],
        "previous_raw_id": 698,
    },
    {
        "doc_type": "invoice",
        "file_path": "documents/invoice/DESIGN HOUSE AGENCY INV DHA-2025-143 for PO438295.pdf",
        "previously_missing": ["invoice_id"],
        "previous_raw_id": 1401,
    },
    {
        "doc_type": "quote",
        "file_path": "documents/quote/INFOTECH QUT25-304-34.pdf",
        "previously_missing": ["quote_id", "quote_date", "total_amount_incl_tax"],
        "previous_raw_id": 42,
    },
]

EXPECTED_BY_DOCTYPE = {
    "invoice": [
        "invoice_id", "invoice_date", "currency", "invoice_amount",
        "tax_amount", "supplier_name", "buyer_id",
    ],
    "purchase_order": [
        "po_id", "order_date", "currency", "total_amount",
        "tax_amount", "supplier_name", "buyer_id",
    ],
    "quote": [
        "quote_id", "quote_date", "currency", "total_amount",
        "tax_amount", "total_amount_incl_tax", "supplier_name",
    ],
}

RAW_TABLE = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
}


def _read_raw_state(conn, doc_type: str, raw_id: int) -> dict:
    cols = EXPECTED_BY_DOCTYPE[doc_type]
    table = RAW_TABLE[doc_type]
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(cols)}, doc_pk_candidate, promotion_status "
            f"FROM {table} WHERE raw_id = %s",
            (raw_id,),
        )
        row = cur.fetchone()
    if row is None:
        return {"_missing_row": True}
    state = dict(zip(cols + ["doc_pk_candidate", "promotion_status"], row))
    return state


def _judge_filled(prev: dict, new: dict, fields: list[str]) -> list[str]:
    """Fields that were NULL/empty before but populated now."""
    lifted = []
    for f in fields:
        before = prev.get(f)
        after = new.get(f)
        before_empty = before is None or before == ""
        after_filled = after is not None and after != ""
        if before_empty and after_filled:
            lifted.append(f)
    return lifted


def main() -> int:
    print(f"L3 verify: judge_model={os.environ.get('EXTRACTION_V3_JUDGE_MODEL')}")
    summary = []
    with get_conn() as conn:
        for case in CASES:
            dt = case["doc_type"]
            print(f"\n=== {dt}  prev_raw_id={case['previous_raw_id']}  {case['file_path']} ===")
            prev_state = _read_raw_state(conn, dt, case["previous_raw_id"])
            print(f"  BEFORE: promotion={prev_state.get('promotion_status')} doc_pk={prev_state.get('doc_pk_candidate')}")
            print(f"  BEFORE missing: {case['previously_missing']}")

            try:
                result = dispatch_document(
                    process_monitor_id=None,
                    file_path=case["file_path"],
                    doc_type=dt,
                )
            except Exception as exc:
                print(f"  ERROR: dispatch raised {exc!r}")
                summary.append({
                    "doc_type": dt, "ok": False, "error": str(exc),
                    "file": case["file_path"],
                })
                continue
            new_raw_id = result["raw_id"]
            print(f"  dispatch: status={result['status']} raw_id={new_raw_id} doc_pk={result['doc_pk']}")
            print(f"  dispatch: missing_required={result.get('missing_required')}")

            new_state = _read_raw_state(conn, dt, new_raw_id)
            lifted = _judge_filled(prev_state, new_state, case["previously_missing"])
            print(f"  AFTER fields lifted by L3 judge: {lifted}")
            for f in case["previously_missing"]:
                bv = prev_state.get(f)
                av = new_state.get(f)
                arrow = "—>" if (bv != av) else "=="
                print(f"    {f}: {bv!r} {arrow} {av!r}")

            summary.append({
                "doc_type": dt,
                "file": case["file_path"],
                "prev_status": prev_state.get("promotion_status"),
                "new_status": result["status"],
                "previously_missing": case["previously_missing"],
                "still_missing": result.get("missing_required") or [],
                "fields_lifted": lifted,
                "new_raw_id": new_raw_id,
            })

    print("\n===== SUMMARY =====")
    for s in summary:
        print(s)

    n_total = len(summary)
    n_promoted = sum(1 for s in summary if s.get("new_status") == "promoted")
    n_lift_any = sum(1 for s in summary if s.get("fields_lifted"))
    print(f"\nDocs: {n_total}   newly promoted: {n_promoted}   any field lift: {n_lift_any}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
