"""Backfill proc.bp_purchase_order_{raw,stg}.quote_reference on POs already extracted.

New uploads pick the field up automatically: it is declared in
extraction_schemas/purchase_order.yaml and both persistence and raw->stg
promotion are db_column-driven. Rows extracted BEFORE the column existed have no
value, so award detection stays blind on them until this runs.

The value is read back out of the row's own stored parser snapshot
(bp_purchase_order_raw.parser_snapshot->>'full_text'), using the SAME compiled
patterns the live extractor uses -- imported from PatternRegistry, never
re-typed here, so the two cannot drift.

NOTHING IS INVENTED. A value is written only when it appears literally in that
document's parsed text; anything else is skipped and counted. POs that cite no
quote (a works package raised against a tender with no bid on file, say) keep a
NULL -- that is the correct answer, not a gap to fill.

Usage:
    python -m scripts.backfill_po_quote_reference            # dry run, prints plan
    python -m scripts.backfill_po_quote_reference --apply    # write
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.extraction.pattern_extractor import _emit_pattern_hits  # noqa: E402
from src.services.extraction.pattern_registry import PatternRegistry  # noqa: E402

FIELD = "quote_reference"


def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
        port=os.environ["DB_PORT"])


def extract_quote_reference(full_text: str) -> str | None:
    """Highest-prior pattern hit for quote_reference, or None.

    Grounding guard: the returned value must be a literal substring of
    ``full_text``. _emit_pattern_hits guarantees this by construction (it slices
    the captured group out of the text), and we re-assert it because this
    function writes to the database.
    """
    if not full_text:
        return None
    registry = PatternRegistry("purchase_order")
    best = None
    for pat in registry.patterns_for(FIELD):
        for cand in _emit_pattern_hits(full_text, pat, None):
            if best is None or cand.confidence > best.confidence:
                best = cand
    if best is None:
        return None
    if best.value not in full_text:          # never write an ungrounded value
        return None
    return best.value


def backfill(apply: bool = False) -> dict:
    conn = _connect()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """select po_id, source_file, parser_snapshot
             from proc.bp_purchase_order_raw
            where quote_reference is null
              and parser_snapshot is not null
            order by po_id""")
    rows = cur.fetchall()

    found, skipped, written = [], [], 0
    for r in rows:
        snap = r["parser_snapshot"] or {}
        text = snap.get("full_text") if isinstance(snap, dict) else None
        ref = extract_quote_reference(text or "")
        if ref is None:
            skipped.append(r["po_id"])
            continue
        found.append((r["po_id"], ref))
        if apply:
            # _raw and _stg are updated together: promotion has already run for
            # these rows, so writing only _raw would leave _stg (what clustering
            # reads) still blind.
            cur.execute("update proc.bp_purchase_order_raw set quote_reference=%s "
                        "where po_id=%s and quote_reference is null", (ref, r["po_id"]))
            cur.execute("update proc.bp_purchase_order_stg set quote_reference=%s "
                        "where po_id=%s and quote_reference is null", (ref, r["po_id"]))
            written += cur.rowcount

    if apply:
        conn.commit()
    conn.close()
    return {"examined": len(rows), "found": found, "no_reference_cited": skipped,
            "applied": apply, "stg_rows_written": written}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    out = backfill(apply=args.apply)
    print(f"examined {out['examined']} PO(s) with no quote_reference\n")
    for po_id, ref in out["found"]:
        print(f"  {po_id}  ->  {ref!r}")
    if out["no_reference_cited"]:
        print(f"\n  cites no quote (left NULL): {', '.join(out['no_reference_cited'])}")
    print(f"\n{'APPLIED' if out['applied'] else 'DRY RUN'} — "
          f"{len(out['found'])} recoverable, {out['stg_rows_written']} _stg row(s) written")
    if not out["applied"]:
        print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
