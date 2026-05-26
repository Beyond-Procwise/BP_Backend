# scripts/extraction_completeness_report.py
"""Report _stg documents whose line items don't reconcile to the header total,
or that have no line items where the doc type expects them. Read-only.

Run: .venv/bin/python scripts/extraction_completeness_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.completeness import assess  # noqa: E402

SPEC = {
    "invoice": ("proc.bp_invoice_stg", "invoice_id", "invoice_amount",
                "proc.bp_invoice_line_items_stg", "line_amount"),
    "purchase_order": ("proc.bp_purchase_order_stg", "po_id", "total_amount",
                       "proc.bp_po_line_items_stg", "line_total"),
    "quote": ("proc.bp_quote_stg", "quote_id", "total_amount",
              "proc.bp_quote_line_items_stg", "line_total"),
}


def main() -> int:
    flagged = 0
    with get_conn() as conn:
        for dt, (htbl, pk, subcol, ltbl, amtcol) in SPEC.items():
            with conn.cursor() as cur:
                cur.execute(f"SELECT {pk}, {subcol} FROM {htbl}")
                headers = cur.fetchall()
            print(f"\n== {dt} ({len(headers)} rows) ==")
            for pk_val, subtotal in headers:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT {amtcol} FROM {ltbl} WHERE {pk} = %s", (pk_val,))
                    lines = [{amtcol: r[0]} for r in cur.fetchall()]
                r = assess(dt, {subcol: subtotal}, lines,
                           has_line_schema=True, missing_required=[])
                if not r.is_complete:
                    flagged += 1
                    print(f"  [{r.status:18s}] {pk_val}  lines={len(lines)} gaps={r.gaps}")
    print(f"\nTotal flagged: {flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
