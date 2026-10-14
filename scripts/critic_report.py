#!/usr/bin/env python
"""What the critic would have done, for a date range.

The point of shadow mode is that this script can be run before anything is
suppressed. It answers: how many were critiqued, how many would have come off
the page, which test killed them, and what the gap register says to fix first.

Usage:
    ./venv/bin/python scripts/critic_report.py --since 2026-09-01
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env"))

from src.services.db import get_conn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="1970-01-01")
    args = parser.parse_args()

    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT verdict, count(*), coalesce(sum(detector_proposed), 0)
              FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s
             GROUP BY verdict ORDER BY 2 DESC
        """, (args.since,))
        rows = cur.fetchall()
        print(f"\nCRITIQUES since {args.since}")
        print(f"{'verdict':<18} {'count':>7} {'detector value':>18}")
        for verdict, count, value in rows:
            print(f"{verdict:<18} {count:>7} {value:>18,.2f}")
        print(f"{'TOTAL':<18} {sum(r[1] for r in rows):>7} "
              f"{sum(r[2] for r in rows):>18,.2f}")

        cur.execute("""
            SELECT detector_type, count(*),
                   coalesce(sum(detector_proposed) FILTER (WHERE would_have_suppressed), 0)
              FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s AND would_have_suppressed
             GROUP BY detector_type ORDER BY 2 DESC
        """, (args.since,))
        print("\nWOULD HAVE BEEN SUPPRESSED, by detector")
        for detector, count, value in cur.fetchall():
            print(f"  {str(detector):<32} {count:>6}  {value:>16,.2f}")

        cur.execute("""
            SELECT t->>'test' AS test, count(*)
              FROM proc.bp_opportunity_critique c,
                   LATERAL jsonb_array_elements(c.tests) AS t
             WHERE c.critiqued_at >= %s
               AND t->>'result' = 'INVALIDATE'
             GROUP BY 1 ORDER BY 2 DESC
        """, (args.since,))
        print("\nWHICH TEST KILLED IT")
        for test, count in cur.fetchall():
            print(f"  {str(test):<28} {count:>6}")

        cur.execute("""
            SELECT g.what_is_missing, g.owner_hint, g.effort, count(*)
              FROM proc.bp_opportunity_gap g
              JOIN proc.bp_opportunity_critique c USING (critique_id)
             WHERE c.critiqued_at >= %s AND g.blocking
             GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 10
        """, (args.since,))
        print("\nBLOCKING GAPS, most findings first")
        for missing, owner, effort, count in cur.fetchall():
            print(f"  {count:>5} x [{effort or '?':^6}] {str(owner):<18} {missing}")

        cur.execute("""
            SELECT count(*) FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s AND shadowed
        """, (args.since,))
        print(f"\nshadowed (recorded, nothing suppressed): {cur.fetchone()[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
