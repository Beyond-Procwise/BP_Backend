"""Re-read the line items of stored documents whose lines do not add up to their total.

The table extractor could not read staffing tables ("Role / Grade | Days | Day rate"),
took the "Detail" column beside a named "Provision" column as the description, and kept
section subtotal rows, so its lines never reconciled and the AI fallback replaced them --
dropping lump-sum lines such as Overtime and Expenses on some versions of a quote. Fixed
2026-10-08 in engineered/table_extractor.py; this applies the fix to documents already stored.

For each quote / purchase order / invoice in _trgt whose line sum differs from its header
total, the source document is parsed again and its lines re-read by the table extractor.
The stored lines are replaced ONLY when the re-read lines add up to the header total (within
1.00) -- the document's own total is the proof. Anything else is left exactly as it is and
listed. _raw is never touched (permanent); _stg lines are rewritten and copied to _trgt.

    python scripts/reread_unreconciled_lines.py            # dry run: what would change
    python scripts/reread_unreconciled_lines.py --apply
"""
from __future__ import annotations

import argparse
import logging

from src.services.db import get_conn
from src.services.extraction import completeness
from src.services.extraction.engineered.table_extractor import extract_line_items
from src.services.extraction.parser import parse
from src.services.extraction.pattern_registry import get_registry
from src.services.extraction.persistence import build_line_items
from src.services.linking_engine import _copy_lines, _rows

DOCS = {
    # doc_type: (header _trgt, pk, header total, raw, lines stg, lines trgt, line no col, line id col)
    "quote": ("proc.bp_quote_trgt", "quote_id", "total_amount", "proc.bp_quote_raw",
              "proc.bp_quote_line_items_stg", "proc.bp_quote_line_items_trgt", "line_number", "quote_line_id"),
    "purchase_order": ("proc.bp_purchase_order_trgt", "po_id", "total_amount", "proc.bp_purchase_order_raw",
                       "proc.bp_po_line_items_stg", "proc.bp_po_line_items_trgt", "line_number", "po_line_id"),
    "invoice": ("proc.bp_invoice_trgt", "invoice_id", "invoice_amount", "proc.bp_invoice_raw",
                "proc.bp_invoice_line_items_stg", "proc.bp_invoice_line_items_trgt", "line_no", "invoice_line_id"),
}
# Columns that describe the LINE; everything else on an existing line row (deal, currency,
# po_id...) is carried onto the re-read lines when it was the same on every old line.
LINE_COLS = {"item_id", "item_description", "quantity", "unit_of_measure", "unit_price", "line_total",
             "line_amount", "tax_percent", "tax_amount", "total_amount", "total_amount_incl_tax",
             "quote_number", "delivery_date", "created_date", "last_modified_date"}


def _unreconciled(cur, doc_type):
    head, pk, tot, raw, stg, trgt, _n, _id = DOCS[doc_type]
    amt = completeness._LINE_AMOUNT_COL[doc_type]
    return _rows(cur, f"""
        select h.{pk} as pk, h.{tot} as total, s.sm as line_sum,
               (select pm.file_path from {raw} r join proc.process_monitor pm on pm.id = r.process_monitor_id
                 where r.{pk} = h.{pk} order by r.raw_id desc limit 1) as file_path
          from {head} h join (select {pk}, sum({amt}) sm from {trgt} group by 1) s using ({pk})
         where h.{tot} is not null and abs(h.{tot} - s.sm) > {completeness._RECONCILE_ABS_TOLERANCE}
         order by 1""")


def _reread(doc_type, file_path):
    reg = get_registry(doc_type)
    return build_line_items(extract_line_items(parse(file_path), reg.schema), reg)


def _rewrite(cur, doc_type, pk_val, lines):
    head, pk, tot, raw, stg, trgt, num_col, id_col = DOCS[doc_type]
    old = _rows(cur, f"select * from {stg} where {pk} = %s", (pk_val,))
    carry = {}
    if old:
        for col in old[0]:
            if col in LINE_COLS or col in (pk, num_col, id_col):
                continue
            vals = {repr(r.get(col)) for r in old}
            if len(vals) == 1:
                carry[col] = old[0].get(col)
    cur.execute(f"delete from {stg} where {pk} = %s", (pk_val,))
    for i, line in enumerate(lines, 1):
        row = {**carry, **line, pk: pk_val, num_col: i, id_col: f"{pk_val}-L{i}"}
        cols = list(row)
        cur.execute(f"insert into {stg} ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})",
                    [row[c] for c in cols])
    return _copy_lines(cur, pk, pk_val, stg, trgt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="commit (default: dry run)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    fixed, left = [], []
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        for doc_type in DOCS:
            for d in _unreconciled(cur, doc_type):
                label = f"{doc_type} {d['pk']}"
                try:
                    lines = _reread(doc_type, d["file_path"]) if d["file_path"] else []
                except Exception as exc:  # noqa: BLE001 -- one unreadable file must not stop the rest
                    left.append(f"{label}: could not re-read ({str(exc)[:80]})")
                    continue
                new_sum = completeness.line_sum(doc_type, lines)
                if not lines or new_sum is None or not completeness._reconciles(new_sum, float(d["total"])):
                    left.append(f"{label}: total {d['total']}, stored lines {d['line_sum']}, "
                                f"re-read {len(lines)} lines summing {new_sum} -- left unchanged")
                    continue
                n = _rewrite(cur, doc_type, d["pk"], lines)
                fixed.append(f"{label}: {d['line_sum']} -> {new_sum:.2f} (= total {d['total']}), {n} lines")
        print("REPLACED (re-read lines add up to the document total):")
        for f in fixed:
            print("  " + f)
        print("LEFT UNCHANGED:")
        for f in left:
            print("  " + f)
        if args.apply:
            conn.commit()
            print(f"committed: {len(fixed)} documents")
        else:
            conn.rollback()
            print("dry run: rolled back (pass --apply to commit)")


if __name__ == "__main__":
    main()
