"""Give a supplier to documents whose supplier name the old name guard wrongly rejected.

supplier_resolver matched bank words as substrings, so "Swift Distribution Partners Ltd"
was rejected as "noise_token" (swift, as in SWIFT/BIC) and its quotes and purchase orders
were stored with no supplier (fixed 2026-10-08). Every rejection is logged in
proc.bp_supplier_name_reject. For each logged name the CURRENT guard accepts, this resolves
the supplier the normal way (alias, exact, fuzzy, or create) and fills supplier_id on the
_stg and _trgt rows of documents that carry that name in _raw and have no supplier yet.
A supplier already set is never changed; _raw is never touched.

    python scripts/reresolve_rejected_suppliers.py            # dry run
    python scripts/reresolve_rejected_suppliers.py --apply
"""
from __future__ import annotations

import argparse
import logging

from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as sr
from src.services.linking_engine import _rows, _table_columns

DOCS = {
    "quote": ("quote_id", "proc.bp_quote_raw", "proc.bp_quote_stg", "proc.bp_quote_trgt"),
    "purchase_order": ("po_id", "proc.bp_purchase_order_raw", "proc.bp_purchase_order_stg",
                       "proc.bp_purchase_order_trgt"),
    "invoice": ("invoice_id", "proc.bp_invoice_raw", "proc.bp_invoice_stg", "proc.bp_invoice_trgt"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="commit (default: dry run)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        names = [r["extracted_name"] for r in _rows(
            cur, "select distinct extracted_name from proc.bp_supplier_name_reject")]
        now_ok = [n for n in names if n and sr._garbage_reason(n) is None]
        print(f"{len(names)} rejected names logged; {len(now_ok)} accepted by the current guard: {now_ok}")
        changed = 0
        for name in now_ok:
            for doc_type, (pk, raw, stg, trgt) in DOCS.items():
                name_cols = [c for c in ("supplier_id", "supplier_name") if c in _table_columns(cur, raw)]
                if not name_cols:
                    continue
                where = " or ".join(f"r.{c} = %s" for c in name_cols)
                pks = [r["pk"] for r in _rows(
                    cur, f"select distinct r.{pk} as pk from {raw} r where ({where}) and r.{pk} is not null",
                    tuple(name for _ in name_cols))]
                if not pks:
                    continue
                sup_id = sr.resolve_or_create_supplier(name, conn, doc_type=doc_type, doc_pk=pks[0])
                if not sup_id:
                    print(f"  {doc_type}: {name!r} still unresolved -- left")
                    continue
                for table in (stg, trgt):
                    cur.execute(f"update {table} set supplier_id = %s where {pk} = any(%s) "
                                f"and (supplier_id is null or supplier_id = '')", (sup_id, pks))
                    if cur.rowcount:
                        changed += cur.rowcount
                        print(f"  {table}: {cur.rowcount} row(s) -> {sup_id}  ({', '.join(map(str, pks))})")
        if args.apply:
            conn.commit()
            print(f"committed: {changed} rows")
        else:
            conn.rollback()
            print(f"dry run: {changed} rows would change; rolled back (pass --apply to commit)")


if __name__ == "__main__":
    main()
