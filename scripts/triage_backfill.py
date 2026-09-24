#!/usr/bin/env python
"""Run discrepancy triage over many deals and print the scale report (spec §9.1).

    set -a; . ./.env; set +a
    ./.venv/bin/python scripts/triage_backfill.py --all
    ./.venv/bin/python scripts/triage_backfill.py --all --dry-run --limit 200
    ./.venv/bin/python scripts/triage_backfill.py --deals DEALV2-005049,DEALV2-000001

--dry-run computes everything and writes nothing. Exit code 1 if any deal failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from src.services.db import get_conn  # noqa: E402
from src.services.triage import loader  # noqa: E402
from src.services.triage.engine import run_triage  # noqa: E402

_NO_DEAL_SQL = "SELECT count(*) FROM proc.bp_invoice_trgt WHERE deal_id IS NULL"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every deal with documents")
    which.add_argument("--deals", help="comma-separated deal ids")
    p.add_argument("--dry-run", action="store_true", help="compute and report; write nothing")
    p.add_argument("--limit", type=int, help="only the first N deals")
    p.add_argument("--report-json", help="also write the report as JSON to this path")
    a = p.parse_args(argv)

    with get_conn() as conn:
        cur = conn.cursor()
        ids = (loader.list_deal_ids(cur) if a.all
               else [d.strip() for d in a.deals.split(",") if d.strip()])
        cur.execute(_NO_DEAL_SQL)
        no_deal = cur.fetchone()[0]
    if a.limit:
        ids = ids[:a.limit]

    def progress(r):
        print(f"  ... {r.deals_done:,}/{r.deals_requested:,} deals, {len(r.failed)} failed",
              flush=True)

    report = run_triage(ids, "backfill", dry_run=a.dry_run, on_batch=progress,
                        known_gaps=[f"Invoices with no deal_id are not triaged: {no_deal:,}"])
    print(report.render())
    if a.report_json:
        with open(a.report_json, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
