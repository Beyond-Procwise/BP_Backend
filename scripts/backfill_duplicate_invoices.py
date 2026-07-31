"""Run the duplicate-invoice detector over the existing corpus.

The scheduler only sees invoices promoted after it was wired in, so this catches up
everything already in proc.bp_invoice_trgt. Dry-run by default: it prints every pair it
would raise, with both documents' references, dates and amounts, the relationship score and
how each signal read, so each one can be checked against the actual documents BEFORE
anything is written. A duplicate-invoice finding accuses a supplier of double-billing — it
should never be created unreviewed.

Each applied duplicate writes TWO rows: the discrepancy on the document (the finding) and a
"Duplicate Invoice Recovery" opportunity on the pipeline (the money to get back). They are
anchored to the same invoice, so the value-summary counts it once.

    set -a; . ./.env; set +a
    PYTHONPATH=.:src ./venv/bin/python scripts/backfill_duplicate_invoices.py          # dry run
    PYTHONPATH=.:src ./venv/bin/python scripts/backfill_duplicate_invoices.py --apply  # write

PYTHONPATH must include src: the FX repository lives there, and without it every non-GBP
recovery is written with a NULL amount (honest, but useless) instead of a converted one.
"""
from __future__ import annotations

import argparse

from src.services.duplicate_invoice_detector import (
    _already_raised, find_duplicates, load_invoices, payment_evidence, run_detector,
)
from src.services.extraction.persistence import get_conn


def _fmt(row: dict) -> str:
    # The currency is printed because these amounts are NATIVE, not GBP: the live corpus
    # bills in INR, USD, AED, GBP and EUR, and adding them up as one number overstates the
    # total by an order of magnitude. /spendiq/value-summary converts; this listing does not.
    total = row.get("total_amount")
    return (f"{row.get('invoice_id')} | {row.get('invoice_date')} | "
            f"{float(total):,.2f} {row.get('currency') or ''} | PO {row.get('po_id') or '—'}")


def _signals(link: dict) -> str:
    """Every signal and how it read, so a pair can be judged on the evidence rather than
    on the score alone."""
    return "  ".join(f"{s['id']}={s['status']}" for s in link.get("signals", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the findings (default is a dry run that writes nothing)")
    args = ap.parse_args()

    with get_conn() as conn:
        cur = conn.cursor()
        dups = find_duplicates(load_invoices(cur))
        fresh = [d for d in dups if not _already_raised(cur, str(d["later"]["invoice_id"]))]

        print(f"{len(dups)} duplicate candidate(s); {len(fresh)} not already recorded")
        print("amounts below are in each invoice's OWN currency — do not sum them; "
              "GET /spendiq/value-summary reports the converted GBP total\n")
        for d in fresh:
            print(f"  supplier : {d['later'].get('supplier_name')}")
            print(f"  earlier  : {_fmt(d['earlier'])}")
            print(f"  later    : {_fmt(d['later'])}   <- would be flagged")
            print(f"  amount   : {d['amount']:,.2f}")
            print(f"  score    : {d['score']:.1f}/100 ({d['band']})")
            print(f"  signals  : {_signals(d['link'])}")
            print(f"  recovery : {payment_evidence(d['later'])[1]}\n")

        if not args.apply:
            print("dry run — nothing written. Re-run with --apply once every pair above "
                  "has been checked against the real documents.\n"
                  "The scheduler's automatic run is OFF until "
                  "DUPLICATE_INVOICE_DETECTOR_ENABLED=1, for the same reason: it scans the "
                  "whole corpus, so its first run writes this entire backlog at once.")
            return 0

        written = run_detector(conn)
        print(f"wrote {written} finding(s)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
