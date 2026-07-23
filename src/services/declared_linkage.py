"""Human-declared connections clustering must never touch.

Precedence tier 1 (spec Precedence): a confirmed proposal, an amend targeting an
existing deal, or an explicit upload grouping outranks any inferred correlation.
This identifies those fixed groups so the clustering core sets them aside. Read-only.
"""
from __future__ import annotations


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def declared_groups(cur, batch_deal_id: str):
    """Fixed (doc_type, doc_pk) groups a human has already decided for this batch.

    One frozenset per CONFIRMED proposal for the batch, containing that proposal's
    members. Never inferred. (Explicit upload-time grouping via process_monitor is a
    future extension, not yet implemented here.)
    """
    confirmed = _rows(cur,
        "select m.proposal_id, m.doc_type, m.doc_pk from proc.bp_deal_proposal_member m "
        "join proc.bp_deal_proposal p on p.proposal_id = m.proposal_id "
        "where p.batch_deal_id = %s and p.status = 'confirmed'",
        (batch_deal_id,))
    by_proposal: dict = {}
    for r in confirmed:
        by_proposal.setdefault(r["proposal_id"], set()).add(
            (r["doc_type"], str(r["doc_pk"])))

    return [frozenset(s) for s in by_proposal.values() if s]
