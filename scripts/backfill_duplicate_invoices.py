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

Narrowing (D-29): ``--only pairs`` or ``--only same-reference`` runs one kind of check, and
``--ref NUMBER`` (repeatable) limits it to those invoice numbers. The listing and the write obey
the same filters, so ``--apply`` writes a finding for exactly the cases a dry run lists (and keeps
the matching recovery items in step, as it always has):

    ... backfill_duplicate_invoices.py --only same-reference --ref 01-2024-002           # list one
    ... backfill_duplicate_invoices.py --only same-reference --ref 01-2024-002 --apply   # write one

PYTHONPATH must include src: the FX repository lives there, and without it every non-GBP
recovery is written with a NULL amount (honest, but useless) instead of a converted one.
"""
from __future__ import annotations

import argparse

from src.services.duplicate_invoice_detector import (
    PAIRS, SAME_REFERENCE, _already_raised, _conflict_already_raised, _conflict_note, _conflict_severity,
    _wanted, find_duplicates, find_reference_conflicts, load_invoices, load_reference_documents,
    payment_evidence, run_detector,
)
from src.services.extraction.persistence import get_conn

_ONLY = {"pairs": PAIRS, "same-reference": SAME_REFERENCE}


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the findings (default is a dry run that writes nothing)")
    ap.add_argument("--only", choices=sorted(_ONLY),
                    help="run one kind of check only: 'pairs' (two promoted invoices) or "
                         "'same-reference' (one number on two raw documents)")
    ap.add_argument("--ref", action="append", metavar="INVOICE_NUMBER",
                    help="limit to this invoice number (repeatable; case and spacing ignored)")
    args = ap.parse_args(argv)
    only = _ONLY[args.only] if args.only else None
    refs = set(args.ref) if args.ref else None
    if refs:
        print(f"limited to invoice number(s): {', '.join(sorted(refs))}")

    with get_conn() as conn:
        cur = conn.cursor()
        if only in (None, PAIRS):
            dups = [d for d in find_duplicates(load_invoices(cur)) if _wanted(d["later"]["invoice_id"], refs)]
        else:
            dups = []
            print("pair scan skipped (--only same-reference)")
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

        # One invoice number on two different documents (read from proc.bp_invoice_raw).
        if only in (None, SAME_REFERENCE):
            conflicts = [c for c in find_reference_conflicts(load_reference_documents(cur)) if _wanted(c["ref"], refs)]
        else:
            conflicts = []
            print("same-number check skipped (--only pairs)")
        fresh_conflicts = [c for c in conflicts if not _conflict_already_raised(cur, str(c["ref"]))]
        print(f"{len(conflicts)} same-number conflict(s); {len(fresh_conflicts)} not already recorded\n")
        for c in fresh_conflicts:
            print(f"  invoice  : {c['ref']}   ({c['kind'].replace('_', ' ')}, {_conflict_severity(c)})")
            for d in c["documents"]:
                flag = "   <- would be flagged" if d is c["later"] else ""
                print(f"  document : {d.get('source_file')} | {d.get('supplier_name')} | "
                      f"{float(d['total_amount']):,.2f} {d.get('currency') or ''} | first read raw_id "
                      f"{d.get('first_raw_id')}{flag}")
            basis = ("the lower total" if c["difference"] is not None
                     else "the later document's own total; the currencies differ")
            print(f"  at risk  : {c['amount']:,.2f} {c['currency'] or ''} ({basis})")
            print(f"  note     : {_conflict_note(c)}\n")

        if not args.apply:
            print("dry run — nothing written. Re-run with --apply once every pair above "
                  "has been checked against the real documents.\n"
                  "The scheduler's automatic run is OFF until "
                  "DUPLICATE_INVOICE_DETECTOR_ENABLED=1, for the same reason: it scans the "
                  "whole corpus, so its first run writes this entire backlog at once.")
            return 0

        written = run_detector(conn, only=only, refs=refs)
        print(f"wrote {written} finding(s)")
        return 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    raise SystemExit(main())
