"""Backfill missing COMPUTABLE columns on _stg rows.

Policy (user, 2026-05-26): if a computable value (USD conversion, tax_amount,
total-incl-tax) is missing on a row but can be derived from the columns that ARE
present, compute it accurately and write it to _stg. (Genuine document
miscalculations are NOT touched here — those are flagged in the discrepancy
table by the pipeline.)

This applies the SAME arithmetic the live pipeline uses
(context_layer._compute_derived), and only fills NULLs — it never overwrites a
value that was extracted/grounded from the document.

Read-mostly (only UPDATEs NULL computable columns). Run:
    .venv/bin/python scripts/backfill_computed_columns.py            # apply
    .venv/bin/python scripts/backfill_computed_columns.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.context_layer import _compute_derived  # noqa: E402

# doc_type -> (stg table, pk col, incl-tax column name)
SPEC = {
    "invoice": ("bp_invoice_stg", "invoice_id", "invoice_total_incl_tax"),
    "purchase_order": ("bp_purchase_order_stg", "po_id", "total_amount_incl_tax"),
    "quote": ("bp_quote_stg", "quote_id", "total_amount_incl_tax"),
}
# Columns we are allowed to fill when NULL (purely computable).
_FILLABLE_BASE = ["tax_amount", "exchange_rate_to_usd", "converted_amount_usd"]


def _num(v):
    return float(v) if isinstance(v, Decimal) else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_filled = 0
    with get_conn() as conn:
        for doc_type, (tbl, pk, incl_col) in SPEC.items():
            fillable = _FILLABLE_BASE + [incl_col]
            with conn.cursor() as cur:
                cur.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema='proc' and table_name=%s", (tbl,))
                present = {r[0] for r in cur.fetchall()}
                fillable = [c for c in fillable if c in present]
                cur.execute(f"select * from proc.{tbl}")
                cols = [d.name for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

            print(f"\n== {doc_type} ({len(rows)} rows) ==")
            for row in rows:
                # Build the input dict the pipeline computes over (numeric coercion).
                src = {k: _num(v) for k, v in row.items()}
                computed = _compute_derived(dict(src))
                updates = {}
                for c in fillable:
                    if row.get(c) in (None, "") and computed.get(c) not in (None, ""):
                        updates[c] = round(float(computed[c]), 4)
                if not updates:
                    continue
                total_filled += 1
                pkv = row[pk]
                desc = ", ".join(f"{c}={v}" for c, v in updates.items())
                print(f"  [{pkv}] fill {desc}")
                if not args.dry_run:
                    set_clause = ", ".join(f"{c} = %s" for c in updates)
                    vals = list(updates.values()) + [pkv]
                    with conn.cursor() as cur:
                        cur.execute(
                            f"update proc.{tbl} set {set_clause} where {pk} = %s", vals)

    action = "would fill" if args.dry_run else "filled"
    print(f"\n{action} computable columns on {total_filled} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
