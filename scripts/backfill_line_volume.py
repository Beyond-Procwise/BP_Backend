"""Fill volume / volume_unit on lines already stored, from their descriptions.

deploy/sql/2026-10-09_line_volume.sql added the columns; extraction fills them from now on
(line_volume.add_line_volumes in dispatch). This applies the same reader to the lines that were
read before it, in the raw, _stg and _trgt line tables of quotes, POs and invoices. Only rows
whose volume is still NULL and whose description names exactly one volume are written; nothing
else on the row changes.

    ./venv/bin/python scripts/backfill_line_volume.py            # dry run
    ./venv/bin/python scripts/backfill_line_volume.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.line_volume import volume_from_description  # noqa: E402

# table -> its row key
TABLES = {
    "proc.bp_quote_line_items_raw": "line_raw_id", "proc.bp_quote_line_items_stg": "quote_line_id",
    "proc.bp_quote_line_items_trgt": "quote_line_id",
    "proc.bp_po_line_items_raw": "line_raw_id", "proc.bp_po_line_items_stg": "po_line_id",
    "proc.bp_po_line_items_trgt": "po_line_id",
    "proc.bp_invoice_line_items_raw": "line_raw_id", "proc.bp_invoice_line_items_stg": "invoice_line_id",
    "proc.bp_invoice_line_items_trgt": "invoice_line_id",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    total = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for table, key in TABLES.items():
            cur.execute(f"SELECT {key}, item_description FROM {table} "
                        f"WHERE volume IS NULL AND item_description ~* '[0-9]'")
            todo = [(k, *volume_from_description(d)) for k, d in cur.fetchall()]
            todo = [(k, v, u) for k, v, u in todo if v is not None]
            print(f"{table}: {len(todo)} lines name a volume")
            total += len(todo)
            if args.apply:
                for k, v, u in todo:
                    cur.execute(f"UPDATE {table} SET volume = %s, volume_unit = %s "
                                f"WHERE {key} = %s AND volume IS NULL", (v, u, k))
    print(f"{'filled' if args.apply else 'would fill'} {total} lines" + ("" if args.apply else " — dry run, pass --apply"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
