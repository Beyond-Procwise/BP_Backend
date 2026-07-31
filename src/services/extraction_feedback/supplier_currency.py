"""What currency this supplier actually bills in, learned from corrections.

The supplier master has a default_currency, but it describes the supplier in general. A
correction describes THIS supplier's invoices, which is better evidence — and it is the
difference between stopping a buyer four times and stopping them three.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# Three independent people agreeing is a policy, not a coincidence. One is an opinion.
MIN_AGREEMENTS = 3
# Below this share of the votes the supplier is genuinely ambiguous — or really does bill in
# more than one currency — and guessing is what this feature exists to prevent.
MAJORITY = 0.75

# doc_pk on bp_extraction_verdict is the invoice's own natural key (promotion.py's
# _STG_PK["invoice"] = "invoice_id", read straight off _raw before _stg/_trgt exist) — the
# same value bp_invoice_trgt.invoice_id carries once promoted, so this join is a like-for-like
# match rather than a guess. Verified live: both columns are `text`.
_LOAD_SQL = """
    SELECT i.supplier_id, v.corrected_value, v.verdict
      FROM proc.bp_extraction_verdict v
      JOIN proc.bp_invoice_trgt i ON i.invoice_id = v.doc_pk
     WHERE v.field_name = 'currency' AND v.doc_type = 'invoice'
       AND i.supplier_id IS NOT NULL
"""


def learned_currency(rows: list[dict], *,
                     min_agreements: int = MIN_AGREEMENTS) -> dict[str, str]:
    """supplier_id -> the currency corrections have settled on, for suppliers where they have.

    Only CORRECTIONS teach: a confirmation says the value we had was right, which tells us
    nothing about what the currency is when we had nothing at all.
    """
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows or []:
        if row.get("verdict") != "corrected":
            continue
        value = str(row.get("corrected_value") or "").strip().upper()
        supplier = row.get("supplier_id")
        if value and supplier:
            votes[supplier][value] += 1

    out: dict[str, str] = {}
    for supplier, tally in votes.items():
        total = sum(tally.values())
        code, n = max(tally.items(), key=lambda kv: kv[1])
        if n >= min_agreements and (n / total) >= MAJORITY:
            out[supplier] = code
    return out


def default_for(cur, supplier_id: str, *,
                learned: Optional[dict[str, str]] = None) -> Optional[str]:
    """The currency to assume for this supplier: what people have taught us, else the
    supplier master's own default, else nothing.

    ``learned`` lets a caller that has already paid for the (full-table-scan) query pass the
    result straight in instead of repeating it — dispatch.py does this with a 15-minute cache
    keyed the same way Task 4 cached reader accuracy, since this runs once per document that
    reaches a bare "$". Omitted, this runs the live query itself on every call, matching the
    single-call contract this function shipped with.
    """
    if not supplier_id:
        return None
    if learned is None:
        cur.execute(_LOAD_SQL)
        cols = [d[0] for d in (cur.description or [])]
        learned = learned_currency([dict(zip(cols, r)) for r in cur.fetchall()])
    if supplier_id in learned:
        return learned[supplier_id]
    cur.execute("SELECT default_currency FROM proc.bp_supplier WHERE supplier_id = %s",
                (supplier_id,))
    row = cur.fetchone()
    return (row[0] or None) if row else None
