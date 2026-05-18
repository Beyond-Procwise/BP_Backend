#!/usr/bin/env python3
"""One-shot snapshot of extraction pipeline state. Safe to run anytime.

Reads (does not modify) the live DB and prints:
- process_monitor status distribution (bp_sqldb + uicanvas)
- _raw discrepancy queue (active-HITL vs legacy)
- recent throughput (last hour, last 24h)
- last 5 health-check snapshots
- last 5 hallucination violations
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
import psycopg2  # noqa: E402

_RAW_TABLES = {
    "invoice": ("bp_invoice_raw", "bp_invoice_stg", "invoice_id"),
    "purchase_order": ("bp_purchase_order_raw", "bp_purchase_order_stg", "po_id"),
    "quote": ("bp_quote_raw", "bp_quote_stg", "quote_id"),
    "contract": ("bp_contract_raw", "bp_contracts", "contract_id"),
}


def _h(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> None:
    s = Settings()
    bp = psycopg2.connect(host=s.db_host, dbname=s.db_name, user=s.db_user,
                          password=s.db_password, port=s.db_port)
    ui = None
    try:
        ui = psycopg2.connect(host=s.db_host, dbname="uicanvas", user=s.db_user,
                              password=s.db_password, port=s.db_port)
    except Exception:
        ui = None

    bcur = bp.cursor()

    _h("bp_sqldb.process_monitor (live)")
    bcur.execute("SELECT status, COUNT(*) FROM proc.process_monitor GROUP BY status ORDER BY status")
    for r in bcur.fetchall():
        print(f"  {r[0]:22s} {r[1]}")
    bcur.execute("""SELECT COUNT(*) FROM proc.process_monitor
                     WHERE status NOT IN ('Extracted','Extraction_Failed','Archived')""")
    print(f"  non-terminal: {bcur.fetchone()[0]}")

    if ui:
        ucur = ui.cursor()
        _h("uicanvas.process_monitor (UI source)")
        ucur.execute("SELECT status, COUNT(*) FROM proc.process_monitor GROUP BY status ORDER BY status")
        for r in ucur.fetchall():
            print(f"  {r[0]:22s} {r[1]}")
        ucur.execute("""SELECT COUNT(*) FROM proc.process_monitor
                         WHERE status NOT IN ('Extracted','Extraction_Failed','Archived')""")
        print(f"  non-terminal: {ucur.fetchone()[0]}")

    _h("_raw discrepancy queue")
    for cat, (raw_t, stg_t, pk_col) in _RAW_TABLES.items():
        bcur.execute(f"""SELECT
            COUNT(*) FILTER (WHERE r.promotion_status='discrepancy'
                AND EXISTS (SELECT 1 FROM proc.bp_extraction_discrepancy d
                             WHERE d.raw_id=r.raw_id AND d.status='open'
                               AND d.severity='critical' AND d.blocks_promotion=TRUE)) AS active,
            COUNT(*) FILTER (WHERE r.promotion_status='discrepancy') AS total
           FROM proc.{raw_t} r""")
        a, t = bcur.fetchone()
        print(f"  {cat:18s} active-HITL={a:4d}  total={t}")

    _h("Throughput")
    for cat, (raw_t, stg_t, pk_col) in _RAW_TABLES.items():
        bcur.execute(f"""SELECT COUNT(*) FROM proc.{stg_t}
                          WHERE COALESCE(created_date, NOW()) > NOW() - INTERVAL '1 hour'""")
        h = bcur.fetchone()[0]
        bcur.execute(f"""SELECT COUNT(*) FROM proc.{stg_t}
                          WHERE COALESCE(created_date, NOW()) > NOW() - INTERVAL '24 hours'""")
        d = bcur.fetchone()[0]
        bcur.execute(f"SELECT COUNT(*) FROM proc.{stg_t}")
        t = bcur.fetchone()[0]
        print(f"  {cat:18s} last_hour={h:4d}  last_24h={d:5d}  all_time={t}")

    _h("Recent health-check snapshots")
    bcur.execute("""SELECT to_regclass('proc.bp_extraction_health_metrics')""")
    if bcur.fetchone()[0] is None:
        print("  (health metrics table not yet populated)")
    else:
        bcur.execute("""SELECT recorded_at, invoice_active_hitl, po_active_hitl,
                              quote_active_hitl, stuck_rows_reset, stuck_rows_failed,
                              audit_sample, audit_violations
                         FROM proc.bp_extraction_health_metrics
                         ORDER BY id DESC LIMIT 5""")
        rows = bcur.fetchall()
        if not rows:
            print("  (no snapshots yet — timer hasn't fired)")
        else:
            for r in rows:
                print(f"  {r[0]}  invoice_hitl={r[1]:3d} po_hitl={r[2]:3d} q_hitl={r[3]:3d}  "
                      f"stuck_reset={r[4]} stuck_fail={r[5]} audit={r[6]}/{r[7]}_viol")

    _h("Recent hallucination violations (last 5)")
    bcur.execute("""SELECT to_regclass('proc.bp_extraction_hallucination_audit')""")
    if bcur.fetchone()[0] is None:
        print("  (audit table not yet populated)")
    else:
        bcur.execute("""SELECT audited_at, doc_type, doc_pk, field_path, value
                          FROM proc.bp_extraction_hallucination_audit
                          ORDER BY id DESC LIMIT 5""")
        rows = bcur.fetchall()
        if not rows:
            print("  (no violations recorded — pipeline is clean)")
        else:
            for r in rows:
                print(f"  {r[0]} {r[1]} pk={r[2]!r} field={r[3]!r} value={r[4]!r:40s}")

    bp.close()
    if ui:
        ui.close()


if __name__ == "__main__":
    main()
