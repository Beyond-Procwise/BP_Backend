"""Reconcile the unit vocabulary against every place units actually appear.

The original drift — a hand-typed map covering 7 of 18 canonical values — went
unnoticed for one reason: nothing counted what the normaliser was refusing. A
value it could not map produced UOM_UNMAPPED and vanished. This script is the
counter.

It scans the authoritative product master and the extracted corpus, normalises
every distinct unit string, and records anything unrecognised in
proc.bp_uom_canonical as status='proposed'.

PROPOSE ONLY. Nothing is ever activated automatically. A proposed unit does not
normalise, so an unconfirmed guess can never quietly start resolving and become
indistinguishable from a confirmed unit — which would make confirmation
pointless. Activating one is a human act:

    UPDATE proc.bp_uom_canonical
       SET status='active', dimension='count',
           confirmed_by='<you>', confirmed_at=now()
     WHERE uom_code='pallet';

...followed by adding it to src/services/facts/uom.py, which the agreement test
in tests/services/facts/test_uom_canonical_table.py enforces.

Sources scanned:
  * proc.bp_product_master   -- authoritative, via the uicanvas FDW bridge
  * proc.bp_*_line_items_trgt -- what extraction actually produced

Usage:
    set -a && . ./.env && set +a
    ./venv/bin/python scripts/reconcile_uom_vocabulary.py            # report
    ./venv/bin/python scripts/reconcile_uom_vocabulary.py --apply    # record
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.facts.uom import normalise_uom  # noqa: E402

logger = logging.getLogger("reconcile_uom")

# (label, SQL). Each must yield (unit_text, occurrences).
_SOURCES: List[Tuple[str, str]] = [
    ("bp_product_master", """
        SELECT lower(btrim(unit_of_measure)) AS u, count(*)
          FROM proc.bp_product_master
         WHERE btrim(coalesce(unit_of_measure, '')) <> ''
         GROUP BY 1
    """),
    ("quote_lines", """
        SELECT lower(btrim(unit_of_measure)) AS u, count(*)
          FROM proc.bp_quote_line_items_trgt
         WHERE btrim(coalesce(unit_of_measure, '')) <> ''
         GROUP BY 1
    """),
    ("invoice_lines", """
        SELECT lower(btrim(unit_of_measure)) AS u, count(*)
          FROM proc.bp_invoice_line_items_trgt
         WHERE btrim(coalesce(unit_of_measure, '')) <> ''
         GROUP BY 1
    """),
    ("po_lines", """
        SELECT lower(btrim(unit_of_measure)) AS u, count(*)
          FROM proc.bp_po_line_items_trgt
         WHERE btrim(coalesce(unit_of_measure, '')) <> ''
         GROUP BY 1
    """),
]

_UPSERT = """
    INSERT INTO proc.bp_uom_canonical
        (uom_code, status, source, observed_count,
         first_observed_at, last_observed_at, is_billing_basis)
    VALUES (%s, 'proposed', %s, %s, now(), now(), true)
    ON CONFLICT (uom_code) DO UPDATE
       SET observed_count = proc.bp_uom_canonical.observed_count + EXCLUDED.observed_count,
           last_observed_at = now(),
           -- never overwrite a human decision
           source = COALESCE(proc.bp_uom_canonical.source, EXCLUDED.source)
"""


def scan(cur) -> Tuple[Counter, Dict[str, str]]:
    """Return (unrecognised unit -> occurrences, unit -> first source seen)."""
    unknown: Counter = Counter()
    origin: Dict[str, str] = {}
    recognised: Counter = Counter()

    for label, sql in _SOURCES:
        try:
            cur.execute(sql)
        except Exception:
            # A source that does not exist in this database is not a failure —
            # bp_product_master needs the uicanvas bridge, which may not be
            # applied everywhere. Say so rather than dying.
            logger.warning("source %s unavailable, skipping", label)
            continue
        for unit, n in cur.fetchall():
            if normalise_uom(unit).canonical is not None:
                recognised[unit] += n
                continue
            unknown[unit] += n
            origin.setdefault(unit, label)

    logger.info("recognised: %d distinct, %d occurrences",
                len(recognised), sum(recognised.values()))
    return unknown, origin


def run(apply: bool = False) -> Tuple[Counter, Counter]:
    """Returns (awaiting review, already ruled on).

    The split matters. Reporting one undifferentiated total would print the
    same number every run no matter how much reviewing had been done, which
    would make a converging queue look like a stuck one — and the point of the
    queue is that it converges.
    """
    from src.services.db import get_conn

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute("select current_database()")
        logger.info("database: %s (apply=%s)", cur.fetchone()[0], apply)

        unknown, origin = scan(cur)

        # Anything a human already ruled on is settled, not outstanding.
        cur.execute("SELECT uom_code FROM proc.bp_uom_canonical "
                    "WHERE status IN ('active', 'rejected')")
        decided = {r[0] for r in cur.fetchall()}

        outstanding = Counter({u: n for u, n in unknown.items() if u not in decided})
        settled = Counter({u: n for u, n in unknown.items() if u in decided})

        if apply and outstanding:
            for unit, n in outstanding.items():
                cur.execute(_UPSERT, (unit, origin.get(unit), n))
            conn.commit()
        else:
            conn.rollback()

    return outstanding, settled


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="record unrecognised units as 'proposed'")
    args = parser.parse_args()

    outstanding, settled = run(apply=args.apply)

    if settled:
        print(f"\nalready ruled on ({len(settled)} distinct) — not re-proposed")

    if not outstanding:
        print("\nnothing awaiting review — every observed unit is either "
              "active or already rejected")
        return 0

    print(f"\nAWAITING REVIEW ({len(outstanding)} distinct):")
    for unit, n in outstanding.most_common():
        print(f"   {unit!r:34s} x{n}")
    if args.apply:
        print("\nrecorded as 'proposed'. They do NOT normalise until a human "
              "confirms them and adds them to src/services/facts/uom.py.")
    else:
        print("\n(dry run — nothing written; pass --apply to record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
