"""Add commercial terms to stored documents' parser snapshots (from their stored text).

New extractions capture payment terms, validity, surcharges... into
parser_snapshot.commercial_terms (engineered/terms_extractor, 2026-10-08). This reads the same
terms from the full_text already stored on each quote / PO / invoice _raw row and ADDS the key;
nothing already in the snapshot is changed, and rows that already have terms are skipped.

    python scripts/backfill_commercial_terms.py            # dry run
    python scripts/backfill_commercial_terms.py --apply
"""
from __future__ import annotations

import argparse
import json

from src.services.db import get_conn
from src.services.extraction.engineered.terms_extractor import terms_from_markdown

RAWS = ("proc.bp_quote_raw", "proc.bp_purchase_order_raw", "proc.bp_invoice_raw")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="commit (default: dry run)")
    args = ap.parse_args()
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        total = 0
        for raw in RAWS:
            cur.execute(f"select raw_id, parser_snapshot->>'full_text' from {raw} "
                        "where parser_snapshot ? 'full_text' and not parser_snapshot ? 'commercial_terms'")
            n = 0
            for raw_id, text in cur.fetchall():
                terms = terms_from_markdown(text or "")
                if not terms:
                    continue
                cur.execute(f"update {raw} set parser_snapshot = parser_snapshot || %s::jsonb where raw_id = %s",
                            (json.dumps({"commercial_terms": terms}), raw_id))
                n += 1
            print(f"{raw}: {n} rows gain commercial_terms")
            total += n
        if args.apply:
            conn.commit()
            print(f"committed: {total}")
        else:
            conn.rollback()
            print(f"dry run: {total} would change; rolled back (pass --apply to commit)")


if __name__ == "__main__":
    main()
