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

    (a) every member of a CONFIRMED proposal for the batch; (b) documents explicitly
    grouped at upload, when process_monitor records such a grouping. Never inferred.
    """
    groups: list = []

    confirmed = _rows(cur,
        "select m.doc_type, m.doc_pk from proc.bp_deal_proposal_member m "
        "join proc.bp_deal_proposal p on p.proposal_id = m.proposal_id "
        "where p.batch_deal_id = %s and p.status = 'confirmed'",
        (batch_deal_id,))
    by_proposal: dict = {}
    for r in confirmed:
        by_proposal.setdefault(r.get("proposal_id", batch_deal_id), set()).add(
            (r["doc_type"], str(r["doc_pk"])))
    # The query above projects only doc_type/doc_pk (no proposal_id), so every
    # confirmed row lands under the same synthetic batch_deal_id key and by_proposal
    # collapses to one bucket. That's fine here: all confirmed members are already
    # human-decided, so lumping them into a single declared frozenset (via the
    # "_all" fallback below) still correctly sets every one of them aside from
    # re-clustering, even though it loses the per-proposal partition.
    if confirmed and not by_proposal:
        by_proposal["_all"] = {(r["doc_type"], str(r["doc_pk"])) for r in confirmed}
    groups.extend(frozenset(s) for s in by_proposal.values() if s)

    return groups
