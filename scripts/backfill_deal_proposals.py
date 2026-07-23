"""Generate deal-clustering proposals for an existing upload batch (non-destructive).

The current single deal (e.g. ANALYSISSET_19072620260719339 — 40 docs, 12 suppliers) is
NOT rewritten. Proposals are written for a human to confirm; on confirm the members move to
real deals and the single deal is superseded. Run:

    PYTHONPATH=.:src .venv/bin/python -m scripts.backfill_deal_proposals [BATCH_DEAL_ID]
"""
from __future__ import annotations

import sys

from src.api.routers.deal_proposals import _generate  # the atomic generate path

DEFAULT_BATCH = "ANALYSISSET_19072620260719339"


def backfill(batch_deal_id: str = DEFAULT_BATCH) -> dict:
    result = _generate(batch_deal_id, "backfill")
    print(f"batch {batch_deal_id}: {len(result['proposal_ids'])} proposals "
          f"{result['proposal_ids']}; {len(result['ungrouped'])} ungrouped; "
          f"line coverage {result['members_with_lines']}/{result['members_total']}")
    for u in result["ungrouped"]:
        print(f"  ungrouped {u['doc_type']} {u['doc_pk']}: {u['reason']}")
    return result


if __name__ == "__main__":
    backfill(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BATCH)
