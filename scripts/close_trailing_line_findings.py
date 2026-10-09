"""Close open line_missing_numbers findings raised on a document's terms / notes block.

Extraction used to raise one "line has no quantity or price" warning for every heading,
bullet and footer row after a document's priced lines ("Payment terms", "Lead time /
service", "•  3-year MSA.", the supplier footer) — 87 on one three-supplier deal, 405
across the corpus. dispatch.py no longer raises them (two_way_match.missing_number_lines);
this applies the same rule to the findings already open.

A finding is judged on the read that raised it (its raw_id's lines), or the document's
current stored lines when that read kept none. It is closed when
  - the read has a priced line and the flagged line sits after the last of them (a terms /
    notes row), or
  - the flagged line now carries numbers in the document's current lines (a later re-read
    captured them).
A row before or between priced lines, and every line of a document with no priced line
(or no stored lines) at all, stays open: those are real misses, or cannot be judged.
Closed as resolution_action='dismiss', resolved_by=<--tag>, the convention the earlier
automatic closures use; nothing is deleted.

    ./venv/bin/python scripts/close_trailing_line_findings.py            # dry run
    ./venv/bin/python scripts/close_trailing_line_findings.py --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.db import get_conn  # noqa: E402

# doc_type -> (raw lines table, stored lines table, document key, line no column, amount column)
LINES = {
    "quote": ("proc.bp_quote_line_items_raw", "proc.bp_quote_line_items_stg", "quote_id", "line_number", "line_total"),
    "invoice": ("proc.bp_invoice_line_items_raw", "proc.bp_invoice_line_items_stg", "invoice_id", "line_no", "line_amount"),
    "purchase_order": ("proc.bp_po_line_items_raw", "proc.bp_po_line_items_stg", "po_id", "line_number", "line_total"),
}
LINES["po"] = LINES["purchase_order"]
_IDX = re.compile(r"^line_items\[(\d+)\]$")


def priced_flags(cur, doc_type: str, raw_id, doc_pk: str) -> tuple[list[bool], list[bool]]:
    """Per line, in order: is it priced — in the read that raised the finding, and in the
    document's current stored lines."""
    raw, stg, key, num, amt = LINES[doc_type]
    cur.execute(f"SELECT quantity, unit_price, {amt} FROM {raw} WHERE raw_id = %s ORDER BY {num} NULLS LAST", (raw_id,))
    read = [any(v not in (None, 0) for v in r) for r in cur.fetchall()]
    cur.execute(f"SELECT quantity, unit_price, {amt} FROM {stg} WHERE {key} = %s ORDER BY {num} NULLS LAST", (doc_pk,))
    now = [any(v not in (None, 0) for v in r) for r in cur.fetchall()]
    return read or now, now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tag", default="trailing_notes_rule_2026_10_09")
    args = ap.parse_args()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT discrepancy_id, doc_type, raw_id, doc_pk_candidate, field_name "
            "FROM proc.bp_extraction_discrepancy "
            "WHERE status = 'open' AND issue_type = 'line_missing_numbers'"
        )
        rows = cur.fetchall()
        docs = {(str(r[1] or "").lower(), r[2], r[3]) for r in rows}
        close, keep, why_closed = [], defaultdict(int), defaultdict(int)
        for did, dt, raw_id, pk, field in rows:
            dt = str(dt or "").lower()
            m = _IDX.match(field or "")
            if dt not in LINES or not pk or not m:
                keep["not a per-line finding on a known document"] += 1
                continue
            i = int(m.group(1))
            read, now = priced_flags(cur, dt, raw_id, pk)
            priced = [j for j, p in enumerate(read) if p]
            if priced and i > priced[-1]:
                close.append(did); why_closed["after the last priced line"] += 1
            elif i < len(now) and now[i]:
                close.append(did); why_closed["the line now carries numbers"] += 1
            elif not priced:
                keep["no priced line stored for the document"] += 1
            else:
                keep["among the priced lines"] += 1

        print(f"open line_missing_numbers: {len(rows)} on {len(docs)} documents")
        for why, n in sorted(why_closed.items()):
            print(f"  close ({why}): {n}")
        for why, n in sorted(keep.items()):
            print(f"  keep ({why}): {n}")
        if not args.apply:
            print("dry run — pass --apply to close them")
            return 0
        cur.execute(
            "UPDATE proc.bp_extraction_discrepancy SET status = 'resolved', "
            "resolution_action = 'dismiss', resolved_by = %s, resolved_at = now(), "
            "notes = coalesce(notes, '') || %s "
            "WHERE discrepancy_id = ANY(%s) AND status = 'open'",
            (args.tag, " [closed: a terms / notes row after the last priced line, or a line a later read priced]", close),
        )
        print(f"closed {cur.rowcount}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
