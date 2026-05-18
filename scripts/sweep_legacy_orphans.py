#!/usr/bin/env python3
"""Re-extract legacy-orphan _raw rows through the renovation pipeline.

A legacy-orphan _raw row:
  - has promotion_status='discrepancy'
  - has no open critical-blocking discrepancies (so HITL isn't owning it)
  - has NULL flat columns (so it can't be promoted as-is — was written by
    the v4.0.0-hybrid pipeline before the renovation flat-column schema
    existed).

For each such row, this script:
  1. Normalises source_file → relative form (so S3 fallback works)
  2. INSERTs a fresh proc.process_monitor row with status='Completed'
  3. Marks the legacy _raw row's pipeline_version with a suffix so a
     future audit can correlate the legacy row with its replacement.

The watcher picks up the new process_monitor row, the renovation pipeline
extracts, the new _raw row gets flat columns, and the row auto-promotes
to _stg (or lands in discrepancy if it genuinely needs HITL).

Files referenced by source_file are resolved by extraction.parser's
local→S3 fallback at extraction time — no need to copy anything here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
import psycopg2  # noqa: E402

# Absolute path → relative form (so S3 lookup uses documents/<cat>/<file>)
_REPO_ROOT = "/home/muthu/PycharmProjects/BP_Backend/"


def _normalise(src: str) -> str:
    src = (src or "").strip()
    if src.startswith(_REPO_ROOT):
        return src[len(_REPO_ROOT):]
    return src


def _orphans(cur, table: str, pk_field: str) -> list[tuple[int, str, str]]:
    """Return (raw_id, source_file, pipeline_version) for legacy orphans."""
    cur.execute(
        f"""SELECT r.raw_id, r.source_file, r.pipeline_version
              FROM proc.{table} r
             WHERE r.promotion_status='discrepancy'
               AND r.{pk_field} IS NULL
               AND r.parser_snapshot IS NULL
               AND NOT EXISTS (SELECT 1 FROM proc.bp_extraction_discrepancy d
                                WHERE d.raw_id=r.raw_id
                                  AND d.status='open' AND d.severity='critical'
                                  AND d.blocks_promotion=TRUE)
             ORDER BY r.raw_id"""
    )
    return cur.fetchall()


def main(dry_run: bool = False) -> None:
    s = Settings()
    conn = psycopg2.connect(
        host=s.db_host, dbname=s.db_name, user=s.db_user,
        password=s.db_password, port=s.db_port,
    )
    conn.autocommit = True
    cur = conn.cursor()

    groups = [
        ("bp_invoice_raw", "invoice_id", "invoice"),
        ("bp_purchase_order_raw", "po_id", "purchase_order"),
        ("bp_quote_raw", "quote_id", "quote"),
        ("bp_contract_raw", "contract_id", "contract"),
    ]
    total_planned = 0
    total_triggered = 0
    seen_files: set[str] = set()
    for table, pk_field, cat in groups:
        rows = _orphans(cur, table, pk_field)
        print(f"\n[{cat}] {len(rows)} legacy-orphan _raw rows")
        for raw_id, source_file, pv in rows:
            if not source_file:
                print(f"  raw_id={raw_id}: skipped (no source_file)")
                continue
            rel = _normalise(source_file)
            # Skip duplicate triggers in the same run (multiple legacy rows
            # may point at the same source file)
            if rel in seen_files:
                print(f"  raw_id={raw_id}: skipped (duplicate trigger for {rel!r})")
                continue
            seen_files.add(rel)
            total_planned += 1
            if dry_run:
                print(f"  raw_id={raw_id}: WOULD trigger re-extraction of {rel!r}")
                continue
            try:
                cur.execute(
                    """INSERT INTO proc.process_monitor
                         (process_name, type, status, file_path, category,
                          document_type, created_date, lastmodified_date, created_by)
                       VALUES (%s, 'inbound', 'Completed', %s, %s, %s, NOW(), NOW(), %s)
                       RETURNING id""",
                    (f"sweep-legacy-orphan-{raw_id}", rel, cat, cat,
                     f"sweep-legacy-from-raw-{raw_id}"),
                )
                new_pm = cur.fetchone()[0]
                # Annotate the legacy _raw row so an audit can correlate it
                cur.execute(
                    f"UPDATE proc.{table} SET pipeline_version = COALESCE(pipeline_version,'') "
                    f"|| '-superseded-by-pm-' || %s WHERE raw_id=%s",
                    (str(new_pm), raw_id),
                )
                total_triggered += 1
                print(f"  raw_id={raw_id}: triggered pm_id={new_pm} for {rel!r}")
            except Exception as exc:
                print(f"  raw_id={raw_id}: ERROR {exc}")

    print(f"\nsummary: planned={total_planned} triggered={total_triggered}")
    conn.close()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
