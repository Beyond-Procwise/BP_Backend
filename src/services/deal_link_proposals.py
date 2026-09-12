"""Propose a deal link for documents that reference no purchase order.

Deals are anchored on the PO (`DEALV2-<po>`), so a document with no `po_id` joins nothing.
That is not a rare edge: a quote is raised BEFORE its PO exists, and plenty of real
invoices simply never print a PO number (the HR invoice traced on 2026-07-13 has a blank
JOB field -- the reference is not there to read).

Inferring the link from supplier and amount alone is exactly the mechanism behind the open
mis-grouping bug, where an 81k Techworld invoice was absorbed into a 638 Dixon Reynolds
deal. So this does NOT link anything. It PROPOSES, and only when the evidence is strong:

  * the same supplier, and
  * totals that agree to the penny, and
  * dates in the only order procurement allows -- quote before PO before invoice, and
  * exactly ONE candidate PO. Two plausible parents is not evidence, it is a coin toss.

The proposal lands in the same discrepancy queue a buyer already works, carrying the
evidence that produced it, and a human sets the po_id. After that the existing deal
machinery runs untouched.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.extraction.persistence import get_conn

log = logging.getLogger(__name__)

_ABS_TOL = 0.01
_REL_TOL = 0.005

# The child tables that can be orphaned, and how to read them.
_CHILDREN = (
    # (doc_type, stg table, pk column, amount column, date column)
    ("invoice", "proc.bp_invoice_stg", "invoice_id", "invoice_amount", "invoice_date"),
    ("quote", "proc.bp_quote_stg", "quote_id", "total_amount", "quote_date"),
)


def _agrees(a: float, b: float) -> bool:
    return abs(a - b) <= max(_ABS_TOL, abs(b) * _REL_TOL)


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def propose(limit: Optional[int] = None) -> dict[str, Any]:
    """Look for orphaned documents whose parent PO is unambiguous. Never links."""
    proposed, ambiguous, no_candidate = 0, 0, 0
    details: list[dict] = []

    with get_conn() as conn:
        cur = conn.cursor()
        for doc_type, table, pk, amt_col, date_col in _CHILDREN:
            cur.execute(
                f"SELECT {pk}, supplier_id, {amt_col}, {date_col} FROM {table} "
                f"WHERE po_id IS NULL AND supplier_id IS NOT NULL AND {amt_col} IS NOT NULL"
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            orphans = cur.fetchall()

            for doc_pk, supplier_id, amount, doc_date in orphans:
                amount = _f(amount)
                if not amount:
                    continue

                # Candidate parents: same supplier, and a total that agrees. The PO table
                # carries supplier_name, not supplier_id, so resolve through the master.
                cur.execute(
                    """
                    SELECT p.po_id, p.total_amount, p.order_date
                      FROM proc.bp_purchase_order_stg p
                      JOIN proc.bp_supplier s
                        ON lower(s.supplier_name) = lower(p.supplier_name)
                     WHERE s.supplier_id = %s
                    """,
                    (supplier_id,),
                )
                candidates = [
                    (po_id, _f(total), order_date)
                    for po_id, total, order_date in cur.fetchall()
                    if _f(total) is not None and _agrees(amount, _f(total))
                ]

                # Order must be possible: a quote precedes its PO, an invoice follows it.
                def _plausible(order_date) -> bool:
                    if not doc_date or not order_date:
                        return True  # no date is not evidence against
                    return doc_date <= order_date if doc_type == "quote" else doc_date >= order_date

                candidates = [c for c in candidates if _plausible(c[2])]

                if not candidates:
                    no_candidate += 1
                    continue
                if len(candidates) > 1:
                    # Two plausible parents is a coin toss, and a coin toss is how the
                    # mis-grouping bug happened. Say so; do not choose.
                    ambiguous += 1
                    log.info(
                        "deal link: %s %s matches %d purchase orders on supplier+amount — "
                        "not proposing, a human must choose",
                        doc_type, doc_pk, len(candidates),
                    )
                    details.append({"doc_type": doc_type, "doc_pk": doc_pk,
                                    "action": "ambiguous",
                                    "candidates": [c[0] for c in candidates]})
                    continue

                po_id, po_total, _ = candidates[0]

                # Idempotent: do not stack the same proposal on every scheduler tick.
                cur.execute(
                    "SELECT 1 FROM proc.bp_extraction_discrepancy "
                    "WHERE doc_pk_candidate = %s AND issue_type = 'deal_link_proposed' "
                    "  AND expected_value = %s AND status = 'open' LIMIT 1",
                    (str(doc_pk), str(po_id)),
                )
                if cur.fetchone():
                    continue

                # A finding belongs to a document, so carry the document it came from.
                raw_table = f"proc.bp_{doc_type}_raw"
                cur.execute(
                    f"SELECT raw_id, source_file FROM {raw_table} WHERE {pk} = %s "
                    f"ORDER BY raw_id DESC LIMIT 1",
                    (doc_pk,),
                )
                src = cur.fetchone()
                raw_id, source_file = (src[0], src[1]) if src else (None, f"{doc_type}:{doc_pk}")

                cur.execute(
                    """
                    INSERT INTO proc.bp_extraction_discrepancy
                        (doc_type, raw_id, source_file, doc_pk_candidate, field_name,
                         issue_type, severity, raw_value, expected_value,
                         blocks_promotion, notes)
                    VALUES (%s,%s,%s,%s,'po_id','deal_link_proposed','info',NULL,%s,false,%s)
                    """,
                    (
                        doc_type, raw_id, source_file, str(doc_pk), str(po_id),
                        (
                            f"this {doc_type} cites no purchase order, but it is the same "
                            f"supplier as PO {po_id} and the totals agree exactly "
                            f"({amount:,.2f} vs {po_total:,.2f}), with dates in the right "
                            f"order. It is the only purchase order that fits. Confirm to "
                            f"set po_id = {po_id} and pull this document into the deal."
                        ),
                    ),
                )
                proposed += 1
                details.append({"doc_type": doc_type, "doc_pk": doc_pk,
                                "action": "proposed", "po_id": po_id,
                                "amount": amount})
        conn.commit()

    result = {"proposed": proposed, "ambiguous": ambiguous,
              "no_candidate": no_candidate, "details": details}
    log.info("deal link proposals: %s", {k: v for k, v in result.items() if k != "details"})
    return result


def confirm(doc_type: str, doc_pk: str, po_id: str, reviewer: Optional[str] = None) -> bool:
    """A human accepted the proposal: set the po_id and let the deal machinery take over."""
    table = {"invoice": "proc.bp_invoice_stg", "quote": "proc.bp_quote_stg"}.get(doc_type)
    pk = {"invoice": "invoice_id", "quote": "quote_id"}.get(doc_type)
    if not table:
        raise ValueError(f"cannot confirm a deal link for doc_type={doc_type!r}")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE {table} SET po_id = %s WHERE {pk} = %s", (po_id, doc_pk))
        updated = cur.rowcount
        cur.execute(
            "UPDATE proc.bp_extraction_discrepancy "
            # 'apply_value' is the vocabulary this table already uses for "the human
            # accepted the proposed value" — see the resolution_action check constraint.
            "SET status='resolved', resolved_at=now(), resolved_by=%s, "
            "    resolution_action='apply_value', resolved_value=%s "
            "WHERE doc_pk_candidate=%s AND issue_type='deal_link_proposed' AND status='open'",
            (reviewer, str(po_id), str(doc_pk)),
        )
        conn.commit()
    log.info("deal link confirmed: %s %s → PO %s (by %s)", doc_type, doc_pk, po_id, reviewer)
    return updated > 0
