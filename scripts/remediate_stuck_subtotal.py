#!/usr/bin/env python3
"""Remediate documents stuck at promotion_status='discrepancy' because the
required header subtotal (invoice_amount / total_amount) could not be grounded,
even though the line items are present.

Applies the same subtotal-closure-aware derivation the extraction pipeline now
does (completeness.derive_subtotal_from_lines): set the subtotal on the _raw
row, trim mis-captured Subtotal/Tax line rows, resolve the blocking
missing_required discrepancy, and — when no other blocking discrepancy remains —
promote _raw -> _stg. Idempotent; pure arithmetic over grounded line amounts.

Usage:  python3 scripts/remediate_stuck_subtotal.py            # dry-run
        python3 scripts/remediate_stuck_subtotal.py --apply    # commit + promote
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# doc_type -> (raw table, line-items-raw table, subtotal column)
_DOC = {
    "invoice": ("proc.bp_invoice_raw", "proc.bp_invoice_line_items_raw", "invoice_amount"),
    "quote": ("proc.bp_quote_raw", "proc.bp_quote_line_items_raw", "total_amount"),
    "purchase_order": ("proc.bp_purchase_order_raw", "proc.bp_po_line_items_raw", "total_amount"),
}


def _connect():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for pg, db in [("PGHOST", "DB_HOST"), ("PGDATABASE", "DB_NAME"), ("PGUSER", "DB_USER"),
                   ("PGPASSWORD", "DB_PASSWORD"), ("PGPORT", "DB_PORT")]:
        if env.get(db):
            os.environ.setdefault(pg, env[db])
    os.environ.setdefault("PGSSLMODE", "require")
    import psycopg2
    return psycopg2.connect(
        host=os.environ["PGHOST"], dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT", "5432"), sslmode="require")


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(apply: bool):
    from src.services.extraction.completeness import derive_subtotal_from_lines, _LINE_AMOUNT_COL
    conn = _connect()
    conn.autocommit = False
    cur = conn.cursor()
    promoted, fixed, skipped = [], [], []

    for doc_type, (raw_t, line_t, sub_col) in _DOC.items():
        amt_key = _LINE_AMOUNT_COL.get(doc_type, "line_amount")
        # stuck raws with an OPEN blocking missing_required on the subtotal field
        stuck = _rows(cur,
            f"select r.raw_id, r.{sub_col} as sub, r.promotion_status "
            f"from {raw_t} r where r.promotion_status='discrepancy' "
            f"and exists (select 1 from proc.bp_extraction_discrepancy d "
            f"  where d.raw_id=r.raw_id and d.field_name=%s and d.issue_type='missing_required' "
            f"  and d.blocks_promotion=true and d.status='open')",
            (sub_col,))
        for s in stuck:
            raw_id = s["raw_id"]
            if s["sub"] is not None:
                skipped.append((doc_type, raw_id, "subtotal already set"))
                continue
            lines = _rows(cur, f"select line_raw_id, line_no, {amt_key} as amt from {line_t} "
                               f"where raw_id=%s order by line_no", (raw_id,))
            sub, cut = derive_subtotal_from_lines(doc_type, [{amt_key: x["amt"]} for x in lines])
            if sub is None:
                skipped.append((doc_type, raw_id, "no derivable subtotal (no line amounts)"))
                continue

            # 1) set the subtotal on _raw
            cur.execute(f"update {raw_t} set {sub_col}=%s where raw_id=%s", (sub, raw_id))
            # 2) trim mis-captured summary rows (keep first `cut` real items)
            trimmed = 0
            if cut is not None and 0 < cut < len(lines):
                drop_ids = [x["line_raw_id"] for x in lines[cut:]]
                cur.execute(f"delete from {line_t} where line_raw_id = any(%s)", (drop_ids,))
                trimmed = len(drop_ids)
            # 3) resolve the blocking discrepancy
            cur.execute(
                "update proc.bp_extraction_discrepancy set status='resolved', resolved_at=now(), "
                "resolved_value=%s, resolution_action='apply_value', "
                "resolved_by='remediate_stuck_subtotal(derived_subtotal)' "
                "where raw_id=%s and field_name=%s and issue_type='missing_required' and status='open'",
                (str(sub), raw_id, sub_col))
            # 4) any remaining blocking discrepancies?
            remaining = _rows(cur,
                "select count(*) c from proc.bp_extraction_discrepancy "
                "where raw_id=%s and blocks_promotion=true and status='open'", (raw_id,))[0]["c"]
            unblocked = remaining == 0
            if unblocked:
                cur.execute(f"update {raw_t} set promotion_status='pending' where raw_id=%s", (raw_id,))
            fixed.append((doc_type, raw_id, sub, trimmed, "unblocked" if unblocked else f"{remaining} blocking left"))

        # collect raw_ids to promote (unblocked) for this doc_type
    if not apply:
        conn.rollback()
        print("DRY-RUN (no changes written). Use --apply to commit + promote.\n")
    else:
        conn.commit()
        print("COMMITTED _raw subtotal fixes + discrepancy resolutions.\n")

    print("FIXED (doc_type, raw_id, derived_subtotal, lines_trimmed, status):")
    for f in fixed:
        print("  ", f)
    print("SKIPPED:")
    for s in skipped:
        print("  ", s)

    # promote the unblocked ones (own-connection, commits) — only when applying
    if apply:
        from src.services.extraction.promotion import promote
        to_promote = [(dt, rid) for (dt, rid, _sub, _t, st) in fixed if st == "unblocked"]
        print(f"\nPromoting {len(to_promote)} unblocked raw rows _raw -> _stg ...")
        for dt, rid in to_promote:
            try:
                res = promote(rid, dt)
                promoted.append((dt, rid, res.get("ok"), res.get("reason")))
            except Exception as exc:  # noqa: BLE001
                promoted.append((dt, rid, False, repr(exc)[:120]))
        for p in promoted:
            print("  promote:", p)
    conn.close()


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
