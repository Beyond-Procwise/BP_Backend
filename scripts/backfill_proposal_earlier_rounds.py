"""Link earlier quote rounds that deals confirmed before 2026-10-08 left behind.

Proposals then listed only each bid's latest round, so confirming one stranded V1/V2
on no deal or on the upload's batch label. Confirm now gathers them
(proposal_store.attach_earlier_rounds); this applies the same step, with the same
rules, to proposals already confirmed. Rounds another real deal holds are not moved.

    python scripts/backfill_proposal_earlier_rounds.py            # dry run, rolls back
    python scripts/backfill_proposal_earlier_rounds.py --apply
"""
from __future__ import annotations

import argparse

from src.services.db import get_conn
from src.services.proposal_store import _rows, attach_earlier_rounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="commit (default: dry run)")
    args = ap.parse_args()
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        props = _rows(cur, "select proposal_id, batch_deal_id, deal_id, proposed_name "
                           "from proc.bp_deal_proposal where status = 'confirmed' "
                           "and deal_id is not null order by proposal_id")
        total = 0
        for p in props:
            members = _rows(cur, "select doc_type, doc_pk, base_reference, role "
                                 "from proc.bp_deal_proposal_member where proposal_id = %s",
                            (p["proposal_id"],))
            linked = attach_earlier_rounds(cur, p["proposal_id"], members,
                                           deal_id=p["deal_id"], deal_name=p["proposed_name"],
                                           batch_deal_id=p["batch_deal_id"])
            if linked:
                total += len(linked)
                print(f"proposal {p['proposal_id']} -> {p['deal_id']}: {', '.join(linked)}")
        print(f"{len(props)} confirmed proposals checked, {total} earlier rounds linked")
        if args.apply:
            conn.commit()
            print("committed")
        else:
            conn.rollback()
            print("dry run: rolled back (pass --apply to commit)")


if __name__ == "__main__":
    main()
