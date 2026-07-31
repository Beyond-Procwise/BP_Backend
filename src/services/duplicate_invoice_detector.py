"""Duplicate-invoice detector (Value Found, Phase 2).

Paying the same invoice twice is money already out of the door, and unlike a benchmark
opportunity it is *recoverable* — which is why a duplicate lands in the verified tier of
GET /spendiq/value-summary alongside over-billing.

The cost of a false positive here is high: accusing a supplier of double-billing a recurring
monthly charge damages a relationship over nothing. So the rule is deliberately conservative
and demands agreement on every axis at once:

    same supplier (normalised)
    AND the same total, to the penny
    AND a positive total (a credit note is money coming back, never a double payment)
    AND either the same purchase order OR two references one character apart
    AND both dated inside a 90-day window
    AND two genuinely different invoice_ids

A recurring charge fails the fourth test (different PO, unrelated references) and is left
alone. Findings are written to proc.bp_extraction_discrepancy with
issue_type='duplicate_invoice', which value_summary_service already reads — no further
wiring.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Two invoices further apart than this are a repeat purchase, not a double payment.
WINDOW_DAYS = 90


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm_supplier(name: Any) -> str:
    """'Techworld Ltd.' and '  TECHWORLD  LTD ' are one company billing twice."""
    if name is None:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return " ".join(cleaned.split())


def _norm_ref(ref: Any) -> str:
    if ref is None:
        return ""
    return re.sub(r"\s+", "", str(ref).strip().lower())


def _pence(amount: Any) -> Optional[int]:
    """The billed amount in whole pence, or None when there is no amount. Comparing pence
    rather than floats is what makes 'the same total' mean to the penny and not
    approximately."""
    if amount is None:
        return None
    try:
        return int(round(float(amount) * 100))
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _refs_near(a: Any, b: Any) -> bool:
    """True when two references are the same reference, possibly re-keyed.

    Identical after normalisation, or differing by exactly one INSERTED/DELETED character
    ('INV-100' vs 'INV-100A' — the same document re-issued). A substituted character is NOT
    near: 'INV-100' and 'INV-200' are two different invoices that happen to look alike, and
    treating them as one is precisely the false positive this detector must not make. (This
    is narrower than a plain Levenshtein<=1, which would call those two a match.)

    Two absent references agree on nothing, so an empty ref is never near anything.
    """
    x, y = _norm_ref(a), _norm_ref(b)
    if not x or not y:
        return False
    if x == y:
        return True
    if abs(len(x) - len(y)) != 1:
        return False
    longer, shorter = (x, y) if len(x) > len(y) else (y, x)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


_SEQUENCE_SUFFIX = re.compile(r"^(?P<stem>.*?)[-_/ ]?(?P<seq>\d{1,3})$")


def _sequence_pair(a: Any, b: Any) -> bool:
    """True when two references are consecutive members of ONE numbered series —
    'INV000469-1' and 'INV000469-2': the same stem, different trailing sequence numbers.

    This is the normal shape of several invoices billed against one purchase order, and it
    is emphatically NOT a duplicate. It has to be checked because the same-PO branch below
    cannot tell a series apart from a double bill on its own: measured against the live
    corpus (2026-07-31), same-supplier + same-total + same-PO alone flagged 4,974 of 12,408
    invoices — every one of them a member of an -1/-2/-3 series, none of them a duplicate.
    """
    ma, mb = _SEQUENCE_SUFFIX.match(_norm_ref(a)), _SEQUENCE_SUFFIX.match(_norm_ref(b))
    if not ma or not mb:
        return False
    return (ma.group("stem") == mb.group("stem")
            and ma.group("stem") != ""
            and ma.group("seq") != mb.group("seq"))


def _paper_trail_agrees(earlier: dict, later: dict) -> bool:
    """The evidence that these are the same bill and not two similar ones."""
    ref_a, ref_b = earlier.get("invoice_ref"), later.get("invoice_ref")
    # A numbered series is never a duplicate, however well everything else lines up.
    if _sequence_pair(ref_a, ref_b):
        return False
    po_a, po_b = _norm_ref(earlier.get("po_id")), _norm_ref(later.get("po_id"))
    if po_a and po_b and po_a == po_b:
        return True
    return _refs_near(ref_a, ref_b)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def find_duplicates(invoices: list[dict]) -> list[dict]:
    """Pure. Each result is {'later': row, 'earlier': row, 'amount': float} — the LATER
    invoice is the one flagged, because it is the one that should not have been paid, and
    `amount` is its full total (the whole payment is at risk, not a difference).

    Bucketed on (supplier, total): both are mandatory conjuncts of the rule, so grouping by
    them changes nothing about the answer and keeps 12,000 live invoices from becoming 72
    million comparisons.
    """
    buckets: dict[tuple[str, int], list[dict]] = {}
    for inv in invoices or []:
        supplier = _norm_supplier(inv.get("supplier_name"))
        pence = _pence(inv.get("total_amount"))
        when = _as_date(inv.get("invoice_date"))
        iid = inv.get("invoice_id")
        # Absent data is not agreement — a row missing any of these is simply not compared.
        # A non-positive total is a credit note or a nil bill: nothing was paid twice.
        if not supplier or pence is None or pence <= 0 or when is None or not iid:
            continue
        buckets.setdefault((supplier, pence), []).append(inv)

    window = timedelta(days=WINDOW_DAYS)
    out: list[dict] = []
    for rows in buckets.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (_as_date(r["invoice_date"]), str(r["invoice_id"])))
        for i, later in enumerate(rows):
            for earlier in rows[:i]:
                if str(earlier["invoice_id"]) == str(later["invoice_id"]):
                    continue
                if _as_date(later["invoice_date"]) - _as_date(earlier["invoice_date"]) > window:
                    continue
                if not _paper_trail_agrees(earlier, later):
                    continue
                out.append({"later": later, "earlier": earlier,
                            "amount": round(float(later["total_amount"]), 2)})
                # One finding per document, against the earliest invoice it duplicates —
                # three identical invoices raise two findings, not three overlapping pairs.
                break
    return out


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

# invoice_total_incl_tax is what was actually billed and would actually be paid twice;
# invoice_amount (net) is the fallback for rows where the gross was never extracted.
_LOAD_SQL = """
    SELECT i.invoice_id,
           i.po_id,
           COALESCE(s.supplier_name, i.supplier_id) AS supplier_name,
           COALESCE(i.invoice_total_incl_tax, i.invoice_amount) AS total_amount,
           i.invoice_date
      FROM proc.bp_invoice_trgt i
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
     WHERE i.invoice_id IS NOT NULL
"""


def load_invoices(cur) -> list[dict]:
    """The invoices the rule runs over. invoice_ref IS invoice_id here — bp_invoice_trgt
    has no separate reference column, and the id is the reference printed on the document."""
    cur.execute(_LOAD_SQL)
    rows = []
    for invoice_id, po_id, supplier_name, total_amount, invoice_date in cur.fetchall():
        rows.append({"invoice_id": invoice_id, "po_id": po_id,
                     "supplier_name": supplier_name, "total_amount": total_amount,
                     "invoice_date": invoice_date, "invoice_ref": invoice_id})
    return rows


def _already_raised(cur, doc_pk: str) -> bool:
    """True when this document already carries a duplicate_invoice finding — including one
    a human dismissed. A dismissed pair must never come back: re-raising it would overrule
    the person who looked at both documents and said no."""
    cur.execute(
        "SELECT 1 FROM proc.bp_extraction_discrepancy "
        "WHERE doc_pk_candidate = %s AND issue_type = 'duplicate_invoice' LIMIT 1",
        (str(doc_pk),),
    )
    return cur.fetchone() is not None


def _source_of(cur, invoice_id: str) -> tuple[Optional[int], str]:
    """A finding belongs to a document, so carry the raw row it came from."""
    cur.execute(
        "SELECT raw_id, source_file FROM proc.bp_invoice_raw WHERE invoice_id = %s "
        "ORDER BY raw_id DESC LIMIT 1",
        (str(invoice_id),),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, f"invoice:{invoice_id}")


_INSERT_SQL = """
    INSERT INTO proc.bp_extraction_discrepancy
        (doc_type, raw_id, source_file, doc_pk_candidate, field_name,
         issue_type, severity, raw_value, computed_value, blocks_promotion, status, notes)
    VALUES ('invoice', %s, %s, %s, 'invoice_ref',
            'duplicate_invoice', 'critical', %s, %s, false, 'open', %s)
"""


def _note(dup: dict) -> str:
    earlier = dup["earlier"]
    when = _as_date(earlier.get("invoice_date"))
    return (f"possible duplicate of {earlier['invoice_id']}"
            f"{f' ({when:%Y-%m-%d})' if when else ''}: same supplier and amount, "
            f"matching reference")


def run_detector(conn=None) -> int:
    """Find duplicates and record the ones not already recorded. Returns rows written.

    Idempotent: a document that already carries a duplicate_invoice finding is skipped, so
    the scheduler can call this on every promotion event without stacking findings.
    """
    if conn is not None:
        return _run(conn)
    from src.services.extraction.persistence import get_conn
    with get_conn() as own:
        return _run(own)


def _run(conn) -> int:
    cur = conn.cursor()
    dups = find_duplicates(load_invoices(cur))
    written = 0
    for dup in dups:
        later = dup["later"]
        doc_pk = str(later["invoice_id"])
        if _already_raised(cur, doc_pk):
            continue
        raw_id, source_file = _source_of(cur, doc_pk)
        cur.execute(_INSERT_SQL, (
            raw_id, source_file, doc_pk,
            f"{dup['amount']:.2f}",
            # Task 2's convention: a signed computed_value IS the delta.
            f"+{dup['amount']:.2f}",
            _note(dup),
        ))
        written += 1
    conn.commit()
    logger.info("duplicate-invoice detector: %s candidate(s), %s new finding(s)",
                len(dups), written)
    return written
