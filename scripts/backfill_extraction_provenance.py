"""Backfill the provenance declarations for records promoted before they existed.

Promotion now declares two things it previously did silently, both of which leave a value
in _stg that is indistinguishable on screen from one the document actually printed:

  value_derived            a money figure that appears nowhere in the document's text, so
                           it was computed (typically subtotal + tax, or subtotal x
                           tax_percent) rather than read
  date_precision_inferred  a date whose DAY the pipeline supplied because the page gave
                           only a month ("Nov 2024")

Those checks run at promotion, so they cover new arrivals only. This applies exactly the
same two functions to what is already in _trgt, so the existing corpus is as honest as
everything that follows it.

WHAT IT TOUCHES
    Inserts into proc.bp_extraction_discrepancy and nothing else. No stored amount, date,
    status or promotion state is modified anywhere. Every row is severity 'warning' with
    blocks_promotion false: this is provenance, not an error, and the values remain.

SAFE TO RE-RUN
    Skips any (document, field, issue type) already declared, so a second run inserts
    nothing. Dry run unless --commit is passed, and the dry run prints every row it would
    write.

REQUIRES DOCUMENT TEXT
    Each declaration is judged against the page text stored in that record's latest raw
    row (parser_snapshot.full_text). A corpus whose source documents were never parsed —
    a seeded test dataset, for instance — has no text to judge against, and the script
    correctly declares nothing rather than guessing.

    python scripts/backfill_extraction_provenance.py                     # show the plan
    python scripts/backfill_extraction_provenance.py --commit            # apply it
    python scripts/backfill_extraction_provenance.py --database bp_sqldb # another corpus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.extraction.promotion import (  # noqa: E402
    _log_derived_money,
    _log_imprecise_dates,
)

# (promoted table, raw table, primary key) per document type.
SPEC = {
    "invoice": ("proc.bp_invoice_trgt", "proc.bp_invoice_raw", "invoice_id"),
    "quote": ("proc.bp_quote_trgt", "proc.bp_quote_raw", "quote_id"),
    "purchase_order": ("proc.bp_purchase_order_trgt", "proc.bp_purchase_order_raw", "po_id"),
}
DECLARED = ("value_derived", "date_precision_inferred")


class _Collect:
    """Stands in for a cursor so the promotion checks can be run without writing.

    They are used exactly as promotion uses them; this only captures what they would
    insert, so the backfill cannot drift from live behaviour.
    """

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def execute(self, sql, params=None):        # noqa: D102 - cursor stand-in
        if params:
            self.rows.append(params)


def _connect(database: str | None):
    """Connect using the environment's settings, optionally overriding the database.

    Read directly rather than through get_conn() because this needs a transaction it
    controls (get_conn autocommits) and needs to be pointable at another corpus.
    """
    env = {}
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    get = lambda *names, default=None: next(  # noqa: E731
        (os.environ.get(n) or env.get(n) for n in names if os.environ.get(n) or env.get(n)),
        default,
    )
    return psycopg2.connect(
        host=get("DB_HOST", "PGHOST"),
        port=get("DB_PORT", "PGPORT", default="5432"),
        user=get("DB_USER", "PGUSER"),
        password=get("DB_PASSWORD", "PGPASSWORD"),
        dbname=database or get("DB_NAME", "PGDATABASE"),
        connect_timeout=20,
    )


def _full_text(raw_row: dict) -> str:
    ps = raw_row.get("parser_snapshot")
    if isinstance(ps, str):
        try:
            ps = json.loads(ps)
        except Exception:  # noqa: BLE001
            return ""
    return (ps.get("full_text") or "") if isinstance(ps, dict) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", action="store_true",
                    help="write the rows; without this the script only prints them")
    ap.add_argument("--database", help="target database (default: the configured one)")
    args = ap.parse_args()

    conn = _connect(args.database)
    conn.autocommit = False
    cur = conn.cursor()
    dcur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT current_database()")
    print(f"database: {cur.fetchone()[0]}")

    cur.execute(
        """SELECT doc_pk_candidate, field_name, issue_type
             FROM proc.bp_extraction_discrepancy
            WHERE issue_type = ANY(%s)""",
        (list(DECLARED),),
    )
    seen = {tuple(r) for r in cur.fetchall()}
    print(f"already declared: {len(seen)}")

    pending: list[tuple] = []
    dupes = no_text = documents = 0
    for doc_type, (trgt, raw, pk) in SPEC.items():
        # The LATEST raw row per promoted document: superseded re-extractions of the same
        # document must not each contribute a declaration.
        dcur.execute(f"""
            SELECT r.*
              FROM {trgt} t
              JOIN LATERAL (
                SELECT * FROM {raw} rr
                 WHERE rr.{pk} = t.{pk}
                 ORDER BY rr.extracted_at DESC NULLS LAST, rr.raw_id DESC
                 LIMIT 1
              ) r ON TRUE
             WHERE t.{pk} IS NOT NULL
        """)
        for row in dcur.fetchall():
            row = dict(row)
            documents += 1
            text = _full_text(row)
            if not text:
                no_text += 1
                continue
            collector = _Collect()
            _log_derived_money(collector, doc_type, row["raw_id"], text, row)
            _log_imprecise_dates(collector, doc_type, row["raw_id"], text, row)
            for params in collector.rows:
                key = (params[3], params[4], params[8])   # doc_pk, field, issue_type
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                pending.append(params)

    print(f"promoted documents examined                : {documents}")
    print(f"  no page text — nothing can be said about : {no_text}")
    print(f"  already declared, skipped                : {dupes}")
    print(f"  to insert                                : {len(pending)}\n")

    grouped: dict[str, list[tuple]] = {}
    for params in pending:
        grouped.setdefault(params[8], []).append(params)
    for issue, rows in sorted(grouped.items()):
        print(f"  {issue}  ({len(rows)})")
        for params in rows:
            print(f"     {str(params[3])[:24]:24} {params[4]:22} "
                  f"{str(params[7])[:14]:14} {params[11][:74]}")
        print()

    if not args.commit:
        print("DRY RUN — nothing written. Re-run with --commit to apply.")
        conn.rollback()
        conn.close()
        return 0

    for params in pending:
        cur.execute(
            """INSERT INTO proc.bp_extraction_discrepancy
                 (doc_type, raw_id, source_file, doc_pk_candidate,
                  field_name, raw_value, expected_value, computed_value,
                  issue_type, severity, status, notes, blocks_promotion)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            params,
        )
    conn.commit()
    cur.execute(
        """SELECT issue_type, COUNT(*) FROM proc.bp_extraction_discrepancy
            WHERE issue_type = ANY(%s) GROUP BY 1 ORDER BY 1""",
        (list(DECLARED),),
    )
    print(f"committed {len(pending)} row(s). Now present:")
    for issue, n in cur.fetchall():
        print(f"   {issue:28} {n}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
