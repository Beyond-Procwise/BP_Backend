"""Compare a document against the purchase order it references.

The pipeline already knew how to do this and threw the answer away. `linking_engine`
scores an invoice against its PO across po_ref / supplier / amount / line_set signals, and
marks each one MATCH, CONFLICT or MISSING -- then collapses the lot into a single
confidence number used to gate promotion. The CONFLICTs, which are precisely the facts a
buyer needs ("you were billed for a line that is not on the PO"), were never surfaced
anywhere. The Action Centre filled up with the extractor's own arithmetic complaints while
a supplier over-billing against the PO went unreported.

These findings are raised per DOCUMENT (raw_id + source_file), not per business key. That
matters: several documents legitimately carry the same invoice number (an original and its
corrections), and `_stg` keeps only the latest. A discrepancy belongs to the piece of paper
that contains it.

Deliberately NOT flagged: an invoice billing LESS than its PO. Partial and split invoices
are normal procurement, and crying wolf on every one of them is how a check gets ignored.
Over-billing is the asymmetry that costs money.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.services.extraction.persistence import Discrepancy, get_conn
from src.services.linking_engine import _norm_po, _PO_NORM_SQL

log = logging.getLogger(__name__)

# Money agrees if it is within a penny, or within 0.5% for large sums (rounding, FX drift).
_ABS_TOL = 0.01
_REL_TOL = 0.005


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _agrees(a: float, b: float) -> bool:
    return abs(a - b) <= max(_ABS_TOL, abs(b) * _REL_TOL)


def _norm_item(s: Any) -> str:
    """Loose key for matching an invoice line to a PO line."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _match_po_line(desc: Any, po_lines: list[dict]) -> Optional[dict]:
    """Find the PO line this invoice line is billing for, or None if it is not on the PO.

    Descriptions are not written twice the same way. The PO says
    "FORD Focus 1.9TDI, 100 HP, 5 Doors" and the invoice says "FORD Focus 1.9TDI" -- the
    same car. An exact key treats them as different items and reports BOTH an unauthorised
    charge and an unbilled PO line, which is two false accusations from one formatting
    difference. Match on containment first, then on token overlap.
    """
    key = _norm_item(desc)
    if not key:
        return None

    exact = [l for l in po_lines if _norm_item(l.get("item_description")) == key]
    if exact:
        return exact[0]

    # One description contains the other (the common case: PO is more verbose).
    for l in po_lines:
        pk = _norm_item(l.get("item_description"))
        if pk and (pk.startswith(key) or key.startswith(pk) or pk in key or key in pk):
            return l

    # Otherwise the best token overlap, if it is convincing.
    words = set(key.split())
    best, best_score = None, 0.0
    for l in po_lines:
        pw = set(_norm_item(l.get("item_description")).split())
        if not pw or not words:
            continue
        score = len(words & pw) / len(words | pw)
        if score > best_score:
            best, best_score = l, score
    return best if best_score >= 0.5 else None


def _load_po(po_id: str) -> tuple[Optional[dict], list[dict]]:
    """The referenced PO and its lines, from _trgt if promoted else _stg.

    Cited PO numbers arrive in inconsistent formats -- an invoice may say
    'PO502004' while the PO table stores the bare '502004'. Resolve on the
    same canonical PO number the linking engine already uses for
    quote/invoice -> PO joins (_norm_po / _PO_NORM_SQL from linking_engine),
    tolerant of a PO-prefix and separator differences on either side. Falls
    back to an exact match on the raw citation for safety."""
    canonical = _norm_po(po_id)
    cond = _PO_NORM_SQL.format(col="po_id")
    with get_conn() as conn:
        cur = conn.cursor()
        header = None
        for table in ("proc.bp_purchase_order_trgt", "proc.bp_purchase_order_stg"):
            row = None
            try:
                if canonical:
                    cur.execute(
                        f"SELECT po_id, total_amount, currency FROM {table} WHERE {cond} = %s",
                        (canonical,),
                    )
                    row = cur.fetchone()
                if row is None:
                    cur.execute(
                        f"SELECT po_id, total_amount, currency FROM {table} WHERE po_id = %s",
                        (po_id,),
                    )
                    row = cur.fetchone()
            except Exception:  # noqa: BLE001 - table may not exist in some envs
                conn.rollback()
                continue
            if row:
                header = {"po_id": row[0], "total_amount": row[1], "currency": row[2]}
                break
        if header is None:
            return None, []

        # Look up lines under the PO's own stored id -- which is what the canonical
        # match above resolved to, and what the line-items table is keyed by.
        cur.execute(
            "SELECT line_number, item_description, quantity, unit_price, line_total "
            "FROM proc.bp_po_line_items_stg WHERE po_id = %s ORDER BY line_number",
            (header["po_id"],),
        )
        lines = [
            {"line_number": r[0], "item_description": r[1], "quantity": r[2],
             "unit_price": r[3], "line_total": r[4]}
            for r in cur.fetchall()
        ]
    return header, lines


def _po_uploaded_but_unpromoted(po_id: str) -> bool:
    """True when the cited PO exists in the raw tier or as an uploaded file,
    i.e. it reached the system but has not been promoted to _stg/_trgt yet
    (typically held in Discrepancy_Review). Distinguishes "the PO is stuck
    two rows away" from "the supplier cited a PO nobody has ever seen"."""
    canonical = _norm_po(po_id)
    cond = _PO_NORM_SQL.format(col="doc_pk_candidate")
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            if canonical:
                cur.execute(
                    f"SELECT 1 FROM proc.bp_purchase_order_raw WHERE {cond} = %s LIMIT 1",
                    (canonical,),
                )
                if cur.fetchone():
                    return True
        except Exception:  # noqa: BLE001 - raw table may not exist in some envs
            conn.rollback()
        try:
            cur.execute(
                "SELECT 1 FROM proc.process_monitor "
                "WHERE file_path ILIKE %s LIMIT 1",
                (f"%{po_id}%",),
            )
            if cur.fetchone():
                return True
        except Exception:  # noqa: BLE001
            conn.rollback()
    return False


def check_against_po(
    doc_type: str,
    columns: dict[str, Any],
    line_items: list[dict[str, Any]],
) -> list[Discrepancy]:
    """Findings raised by comparing this document with the PO it cites."""
    if doc_type not in ("invoice", "quote"):
        return []

    po_id = columns.get("po_id")
    if not po_id:
        if doc_type == "invoice":
            return [Discrepancy(
                field_name="po_id",
                issue_type="po_reference_missing",
                severity="warning",
                blocks_promotion=False,
                notes="invoice cites no purchase order, so it cannot be three-way matched",
            )]
        return []

    po, po_lines = _load_po(str(po_id).strip())
    if po is None:
        if _po_uploaded_but_unpromoted(str(po_id).strip()):
            return [Discrepancy(
                field_name="po_id",
                issue_type="po_pending_review",
                severity="warning",
                blocks_promotion=False,
                raw_value=str(po_id),
                notes=(
                    f"cites purchase order {po_id}, which was uploaded but is still "
                    f"held in extraction review — the match will be re-checked once "
                    f"that purchase order promotes"
                ),
            )]
        # The single highest-value finding in the set: the supplier has quoted a PO number
        # that does not exist, so nothing downstream can match it to anything.
        return [Discrepancy(
            field_name="po_id",
            issue_type="po_not_found",
            severity="critical",
            blocks_promotion=False,
            raw_value=str(po_id),
            notes=(
                f"cites purchase order {po_id}, which does not exist in the purchase-order "
                f"records — the document cannot be matched to a PO"
            ),
        )]

    out: list[Discrepancy] = []

    # --- header: is the supplier billing MORE than the PO authorised? ---------------
    po_total = _f(po.get("total_amount"))
    doc_total = _f(columns.get("invoice_amount")) or _f(columns.get("total_amount"))
    if po_total and doc_total and doc_total > po_total and not _agrees(doc_total, po_total):
        over = round(doc_total - po_total, 2)
        out.append(Discrepancy(
            field_name="invoice_amount" if doc_type == "invoice" else "total_amount",
            issue_type="amount_over_po",
            severity="critical",
            blocks_promotion=False,
            raw_value=f"{doc_total:.2f}",
            expected_value=f"{po_total:.2f}",
            computed_value=f"+{over:.2f}",
            notes=(
                f"billed {doc_total:,.2f} against a purchase order of {po_total:,.2f} — "
                f"{over:,.2f} more than was authorised"
            ),
        ))

    # --- lines: what is being charged that the PO did not authorise? ----------------
    matched_po_lines: set[int] = set()

    for idx, li in enumerate(line_items or []):
        desc = li.get("item_description")
        key = _norm_item(desc)
        amt = _f(li.get("line_amount")) or _f(li.get("line_total")) or _f(li.get("total_amount"))
        if not key:
            continue

        po_line = _match_po_line(desc, po_lines)
        if po_line is not None:
            matched_po_lines.add(id(po_line))
        if po_line is None:
            out.append(Discrepancy(
                field_name=f"line_items[{idx}]",
                issue_type="line_not_on_po",
                severity="critical",
                blocks_promotion=False,
                raw_value=str(desc)[:200],
                computed_value=(f"{amt:.2f}" if amt is not None else None),
                notes=(
                    f"'{str(desc)[:60]}' is charged on this document but does not appear on "
                    f"purchase order {po['po_id']}"
                ),
            ))
            continue

        po_amt = _f(po_line.get("line_total"))
        if amt is not None and po_amt is not None and amt > po_amt and not _agrees(amt, po_amt):
            out.append(Discrepancy(
                field_name=f"line_items[{idx}]",
                issue_type="line_amount_over_po",
                severity="critical",
                blocks_promotion=False,
                raw_value=f"{amt:.2f}",
                expected_value=f"{po_amt:.2f}",
                computed_value=f"+{round(amt - po_amt, 2):.2f}",
                notes=(
                    f"'{str(desc)[:60]}' billed at {amt:,.2f}; the purchase order says "
                    f"{po_amt:,.2f}"
                ),
            ))

        po_qty, qty = _f(po_line.get("quantity")), _f(li.get("quantity"))
        if qty is not None and po_qty is not None and qty > po_qty:
            out.append(Discrepancy(
                field_name=f"line_items[{idx}].quantity",
                issue_type="line_qty_over_po",
                severity="critical",
                blocks_promotion=False,
                raw_value=f"{qty:g}",
                expected_value=f"{po_qty:g}",
                notes=(
                    f"'{str(desc)[:60]}': {qty:g} billed, {po_qty:g} ordered"
                ),
            ))

    # --- what did the PO authorise that never arrived on the invoice? ---------------
    # Only lines nothing on the document matched to — using the same matcher, so a PO line
    # already accounted for above is never also reported as unbilled.
    for pl in po_lines:
        if id(pl) not in matched_po_lines and pl.get("item_description"):
            out.append(Discrepancy(
                field_name="line_items",
                issue_type="po_line_not_billed",
                severity="info",
                blocks_promotion=False,
                expected_value=str(pl.get("item_description"))[:200],
                notes=(
                    f"purchase order {po['po_id']} includes '{str(pl.get('item_description'))[:60]}', "
                    f"which is not billed on this document (expected on a partial or split invoice)"
                ),
            ))

    return out
