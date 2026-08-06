#!/usr/bin/env python
"""Per-field extraction coverage, read from the extraction record.

Phase 0 finding: unit_of_measure is populated on 99.9% of bp_testdb line rows
and was extracted 12 times in 64,118 provenance records. The column fill rate
is seeded data; the provenance record is what extraction actually produced.
Measure here, never from _trgt NULL counts.

Usage:
    set -a && . ./.env && set +a && ./venv/bin/python scripts/field_coverage.py
    ./venv/bin/python scripts/field_coverage.py --doc-type contract
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.db import get_conn  # noqa: E402

# Collapses line_items[0].x / line_items[7].x into one row.
_SQL = """
WITH totals AS (
    SELECT doc_type, count(DISTINCT doc_pk) AS documents_total
    FROM proc.bp_extraction_provenance_v3
    GROUP BY doc_type
),
per_field AS (
    SELECT doc_type,
           regexp_replace(field_path, '\\[[0-9]+\\]', '[]', 'g') AS field_path,
           count(DISTINCT doc_pk) AS documents_with_field
    FROM proc.bp_extraction_provenance_v3
    GROUP BY 1, 2
)
SELECT p.doc_type, p.field_path, p.documents_with_field, t.documents_total
FROM per_field p
JOIN totals t USING (doc_type)
WHERE (%(doc_type)s::text IS NULL OR p.doc_type = %(doc_type)s)
ORDER BY p.doc_type, p.documents_with_field DESC, p.field_path
"""


def field_coverage(conn: Any, doc_type: Optional[str] = None) -> list[dict]:
    """Coverage per (doc_type, field_path). coverage_pct is None when the
    doc_type has no documents — never 0, which would read as 'extracted nothing'
    rather than 'nothing to extract from'."""
    cur = conn.cursor()
    cur.execute(_SQL, {"doc_type": doc_type})
    cols = [d[0] for d in cur.description]
    out: list[dict] = []
    for row in cur.fetchall():
        rec = dict(zip(cols, row))
        total = rec.get("documents_total") or 0
        rec["coverage_pct"] = (
            round(100.0 * rec["documents_with_field"] / total, 1) if total else None
        )
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-type", default=None)
    args = ap.parse_args()

    with get_conn() as conn:
        rows = field_coverage(conn, args.doc_type)

    if not rows:
        print("no provenance records found")
        return 0

    current = None
    for r in rows:
        if r["doc_type"] != current:
            current = r["doc_type"]
            print(f"\n=== {current} ({r['documents_total']} documents) ===")
        pct = "n/a" if r["coverage_pct"] is None else f"{r['coverage_pct']:5.1f}%"
        print(f"  {pct}  {r['documents_with_field']:6d}  {r['field_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
