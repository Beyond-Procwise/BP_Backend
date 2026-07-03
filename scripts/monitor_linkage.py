#!/usr/bin/env python3
"""Read-only linkage & extraction/promotion health monitor for bp_sqldb.

Snapshots the 3-tier pipeline (raw -> stg -> trgt), deal linkage, and -- most
importantly -- captures the FULL DETAIL of every extraction / promotion issue so
we accumulate an evidence trail for designing a line-item quality gate.

Each run:
  * prints a human-readable summary
  * writes a full structured snapshot to artifacts/linkage_monitor/snapshot_<UTC>.json
  * appends a compact trend line to artifacts/linkage_monitor/history.jsonl

It NEVER writes to the database. Re-run as new documents land.

Usage:  python scripts/monitor_linkage.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "artifacts", "linkage_monitor")

# line-item tables: (label, table, parent_fk, amount_col)
# Inspect BOTH _stg and _trgt: during a stg->trgt promotion cycle _trgt drains,
# so a _trgt-only view reports a false-zero health. _stg shows freshly-extracted
# lines before promotion; _trgt shows what actually landed.
LINE_TABLES = [
    ("invoice/stg", "bp_invoice_line_items_stg", "invoice_id", "line_amount"),
    ("invoice/trgt", "bp_invoice_line_items_trgt", "invoice_id", "line_amount"),
    ("po/stg", "bp_po_line_items_stg", "po_id", "line_total"),
    ("po/trgt", "bp_po_line_items_trgt", "po_id", "line_total"),
    ("quote/stg", "bp_quote_line_items_stg", "quote_id", "line_total"),
    ("quote/trgt", "bp_quote_line_items_trgt", "quote_id", "line_total"),
]


def _load_env() -> dict:
    env: dict = {}
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return env
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _jsonable(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def main() -> int:
    env = _load_env()
    conn = psycopg2.connect(
        host=env.get("DB_HOST") or os.environ.get("PGHOST"),
        dbname=env.get("DB_NAME") or os.environ.get("PGDATABASE"),
        user=env.get("DB_USER") or os.environ.get("PGUSER"),
        password=env.get("DB_PASSWORD") or os.environ.get("PGPASSWORD"),
        port=env.get("DB_PORT") or os.environ.get("PGPORT") or "5432",
    )
    conn.autocommit = True
    cur = conn.cursor()

    def q(sql, args=None):
        cur.execute(sql, args or ())
        cols = [d[0] for d in cur.description] if cur.description else []
        try:
            return cols, cur.fetchall()
        except Exception:
            return cols, []

    def rows_as_dicts(sql, args=None):
        cols, rows = q(sql, args)
        return [{c: _jsonable(v) for c, v in zip(cols, r)} for r in rows]

    def scalar(sql, args=None):
        _, rows = q(sql, args)
        return rows[0][0] if rows else None

    def cols_of(table):
        _, rows = q(
            "select column_name from information_schema.columns "
            "where table_schema='proc' and table_name=%s order by ordinal_position",
            (table,),
        )
        return [r[0] for r in rows]

    ts = datetime.now(timezone.utc)
    snap: dict = {"captured_at": ts.isoformat(), "tier_counts": {}, "line_items": {},
                  "discrepancies": [], "held_in_raw": [], "deal_status": {}, "issues": []}

    # ---- tier counts ----
    print("=" * 70)
    print(f"LINKAGE MONITOR  {ts.isoformat()}")
    print("=" * 70)
    print("\nPIPELINE TIER COUNTS  (raw -> stg -> trgt)")
    for t in ("quote", "purchase_order", "invoice"):
        cells = {}
        for tier in ("raw", "stg", "trgt"):
            try:
                cells[tier] = scalar(f"select count(*) from proc.bp_{t}_{tier}")
            except Exception:
                cells[tier] = None
        snap["tier_counts"][t] = cells
        print(f"  bp_{t:15s} " + "  ".join(f"{k}={v}" for k, v in cells.items()))

    # ---- line-item health + full problematic rows ----
    print("\nLINE-ITEM HEALTH  (per doc-type in _trgt)")
    for label, tbl, fk, amt in LINE_TABLES:
        avail = cols_of(tbl)
        if not avail:
            continue
        # numeric cols to test for "no numbers"
        num_cols = [c for c in ("quantity", "unit_price", amt) if c in avail]
        sel = ["item_description" if "item_description" in avail else "NULL as item_description",
               fk]
        sel += [c for c in ("line_no", "line_number", "quantity", "unit_price", amt) if c in avail]
        full = rows_as_dicts(f"select {', '.join(sel)} from proc.{tbl}")
        total = len(full)
        problems = []
        for r in full:
            desc = (r.get("item_description") or "").strip()
            issues = []
            if r.get(amt) is None:
                issues.append(f"null_{amt}")
            if len(desc) <= 2:
                issues.append("suspect_desc")
            if all(r.get(c) is None for c in num_cols):
                issues.append("no_numbers")
            # arithmetic check
            qy, up = r.get("quantity"), r.get("unit_price")
            la = r.get(amt)
            if qy is not None and up is not None and la is not None:
                try:
                    if abs(float(qy) * float(up) - float(la)) > max(0.02, 0.01 * abs(float(la))):
                        issues.append("arithmetic_mismatch")
                except Exception:
                    pass
            if issues:
                r["_issues"] = issues
                problems.append(r)
                snap["issues"].append({"kind": "line_item", "doc_type": label,
                                       "doc_pk": r.get(fk), "issues": issues, "row": r})
        snap["line_items"][label] = {"total": total, "problem_count": len(problems),
                                     "problems": problems}
        flag = "  <-- CHECK" if problems else ""
        print(f"  {label:8s} lines={total}  problems={len(problems)}{flag}")
        for p in problems:
            print(f"      {p.get(fk)} L{p.get('line_no') or p.get('line_number')}: "
                  f"{(p.get('item_description') or '')[:28]!r} -> {p['_issues']}")

    # ---- full open discrepancies ----
    print("\nOPEN DISCREPANCIES  (full detail)")
    disc = rows_as_dicts(
        "select doc_type, doc_pk_candidate, field_name, raw_value, expected_value, "
        "computed_value, issue_type, severity, blocks_promotion, notes, source_file "
        "from proc.bp_extraction_discrepancy where status='open' "
        "order by blocks_promotion desc, severity"
    )
    snap["discrepancies"] = disc
    if not disc:
        print("  (none open)")
    for d in disc:
        tag = "BLOCKS" if d.get("blocks_promotion") else "warn"
        print(f"  [{tag:6s}] {d.get('doc_type')}/{d.get('doc_pk_candidate')} "
              f"{d.get('field_name')} :: {d.get('issue_type')} ({d.get('severity')})")
        if d.get("notes"):
            print(f"           note: {str(d['notes'])[:90]}")

    # ---- held in raw (with blocking reason) ----
    print("\nHELD IN RAW  (extracted, not promoted to stg)")
    held_any = False
    for t, pk in (("quote", "quote_id"), ("purchase_order", "po_id"), ("invoice", "invoice_id")):
        raw_n = scalar(f"select count(*) from proc.bp_{t}_raw")
        stg_n = scalar(f"select count(*) from proc.bp_{t}_stg")
        if raw_n and stg_n is not None and raw_n > stg_n:
            held_any = True
            stg_pks = {r[0] for r in q(f"select {pk} from proc.bp_{t}_stg")[1]}
            raw_rows = rows_as_dicts(
                f"select {pk}, supplier_id, source_file, promotion_status from proc.bp_{t}_raw")
            for r in raw_rows:
                if r.get(pk) not in stg_pks:
                    blockers = [d for d in disc
                                if d.get("doc_type") == t and d.get("blocks_promotion")
                                and str(d.get("doc_pk_candidate")) == str(r.get(pk))]
                    reason = "; ".join(f"{b['field_name']}:{b['issue_type']}" for b in blockers) or "?"
                    snap["held_in_raw"].append({"doc_type": t, "doc_pk": r.get(pk),
                                                "supplier_id": r.get("supplier_id"),
                                                "source_file": r.get("source_file"),
                                                "blocked_by": reason})
                    print(f"  {t} {r.get(pk)} ({r.get('supplier_id')}) blocked_by: {reason}")
    if not held_any:
        print("  (none)")

    # ---- deal status ----
    print("\nDEAL LINKAGE  (process_monitor status)")
    for status, n in q(
        "select coalesce(status,'(null)'), count(*) from proc.process_monitor "
        "group by 1 order by 2 desc")[1]:
        snap["deal_status"][status] = n
        print(f"  {status:28s} {n}")

    # ---- persist ----
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    snap_path = os.path.join(OUT_DIR, f"snapshot_{stamp}.json")
    with open(snap_path, "w") as f:
        json.dump(snap, f, indent=2, default=str)
    trend = {"captured_at": snap["captured_at"], "tier_counts": snap["tier_counts"],
             "line_problem_counts": {k: v["problem_count"] for k, v in snap["line_items"].items()},
             "open_discrepancies": len(disc),
             "blocking_discrepancies": sum(1 for d in disc if d.get("blocks_promotion")),
             "held_in_raw": len(snap["held_in_raw"]),
             "total_line_issues": len(snap["issues"])}
    with open(os.path.join(OUT_DIR, "history.jsonl"), "a") as f:
        f.write(json.dumps(trend, default=str) + "\n")

    print(f"\nCaptured -> {os.path.relpath(snap_path, ROOT)}  "
          f"(total line issues: {len(snap['issues'])}, held: {len(snap['held_in_raw'])}, "
          f"blocking discrepancies: {trend['blocking_discrepancies']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
