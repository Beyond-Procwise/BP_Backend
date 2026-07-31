"""Duplicate-invoice detection (Value Found, Phase 2).

Paying the same invoice twice is money already out of the door and, unlike a benchmark
opportunity, it is *recoverable* — which is why a duplicate lands in the verified tier of
GET /spendiq/value-summary alongside over-billing.

This does NOT invent its own matching rule. The codebase already has the deterministic
relationship math that decides whether two documents are the same thing — weighted signals
with tiers and conflict caps, a log-odds fusion to an F score, and decision bands
(auto_link / auto_link_with_warning / review / weak_relation / block_or_exception). It is
what links an invoice to its PO and what clusters rival quotes, and it is extensible exactly
the way requirement_similarity registers `quote_rival`. So duplication is one more profile
on that engine — `invoice_duplicate` — and every finding carries the same auditable signal
breakdown as every other link in the product.

Two signals are new, because invoice-to-invoice asks a question PO linkage never does:

  ref_prox   what the two invoice references say about each other. The same reference, or
             the same reference with a re-issue marker appended ("INV-100" / "INV-100A"),
             is the strongest evidence there is; different trailing numbers are evidence
             the other way — the documents number themselves as different members of a set.
  date_prox  how close the two invoice dates are. Same day is a re-issue; months apart is
             a repeat purchase.

Everything else — supplier, purchase order, amount, currency, line set — reuses the
engine's own comparators unchanged.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Optional

from src.services import linking_engine as _le

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The two invoice-to-invoice signals
# ---------------------------------------------------------------------------

_SEQUENCE_SUFFIX = re.compile(r"^(?P<stem>.*?)[-_/ ]?(?P<seq>\d+)$")


def _norm_ref(ref: Any) -> str:
    if ref is None:
        return ""
    return re.sub(r"\s+", "", str(ref).strip().lower())


def _one_edit_apart(x: str, y: str) -> bool:
    """One INSERTED or DELETED character — the appended marker of a re-issue. Not a
    substitution: 'INV-100' and 'INV-200' differ by one character but name two different
    invoices."""
    if abs(len(x) - len(y)) != 1:
        return False
    longer, shorter = (x, y) if len(x) > len(y) else (y, x)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def _sequence_pair(x: str, y: str) -> bool:
    """Two consecutive members of one numbered series — 'INV000469-1' / 'INV000469-2':
    the same stem, different trailing sequence numbers."""
    mx, my = _SEQUENCE_SUFFIX.match(x), _SEQUENCE_SUFFIX.match(y)
    if not mx or not my:
        return False
    return (mx.group("stem") == my.group("stem") != ""
            and mx.group("seq") != my.group("seq"))


def cmp_ref_prox(a: Any, b: Any) -> tuple[float, str]:
    """Reference proximity between two invoices.

    Identical reference is the strongest duplicate evidence there is. One inserted or
    deleted character is a re-issue of the same reference. A numbered series is the
    ordinary shape of several invoices against one purchase order — the documents number
    themselves as different members of a set, which is evidence against, not for.
    Unrelated references say nothing either way.
    """
    x, y = _norm_ref(a), _norm_ref(b)
    if not x or not y:
        return 0.5, "MISSING"
    if x == y:
        return 1.0, "OK"
    # Checked BEFORE the one-edit rule: two references ending in different digit runs are
    # two invoice numbers ("INV-100" / "INV-1000"), not one reference re-keyed. Only a
    # NON-numeric edit — the appended "A" or "-DUP" of a re-issue — reads as the same
    # reference again.
    if _sequence_pair(x, y):
        # A sequence number is positive evidence AGAINST duplication: the documents
        # number themselves as different members of one set. It is a Tier-1 conflict, so
        # the engine's own cap holds the pair down (F <= 60) however perfectly supplier,
        # amount, line set and purchase order agree — which on this corpus they do, the
        # seeder having made every series member byte-identical apart from its number.
        return 0.0, "CONFLICT"
    if _one_edit_apart(x, y):
        return 0.9, "OK"
    return 0.5, "MISSING"


def cmp_date_prox(a: Any, b: Any) -> tuple[float, str]:
    """Invoice-date proximity. Same day is a re-issue; a quarter apart is a repeat
    purchase, not a double payment."""
    da, db = _as_date(a), _as_date(b)
    if da is None or db is None:
        return 0.5, "MISSING"
    days = abs((da - db).days)
    if days == 0:
        return 1.0, "OK"
    if days <= 7:
        return 0.85, "OK"
    if days <= 30:
        return 0.6, "WEAK"
    if days <= 90:
        return 0.35, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("ref_prox", lambda s, t, sl, tl: cmp_ref_prox(s.get("invoice_ref"),
                                                                  t.get("invoice_ref")))
_le.register_signal("date_prox", lambda s, t, sl, tl: cmp_date_prox(s.get("invoice_date"),
                                                                    t.get("invoice_date")))

# Weights mirror what the evidence is actually worth for THIS question. The reference and
# the line set carry the most: two invoices billing identical lines under the same reference
# are the same bill. Supplier and PO are necessary but cheap — thousands of invoices share
# them. Date proximity is a strong discriminator between a re-issue and a repeat purchase,
# so it carries a real conflict cap: dates far apart hold the score down however well
# everything else agrees.
_DUPLICATE_SIGNALS = [
    {"id": "ref_prox",    "cluster": "reference",  "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.60, "kind": "ref_prox"},
    {"id": "supplier_id", "cluster": "identity",   "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.45, "kind": "supplier_id"},
    {"id": "amount",      "cluster": "commercial", "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "amount"},
    {"id": "line_set",    "cluster": "line",       "tier": 2, "weight": 5, "appl": 1.0, "cap": 0.70, "kind": "line_set"},
    # Tier 1: for "was this bill paid twice", the date is not context — two invoices a
    # quarter apart are a repeat purchase, and only a Tier-1 conflict cap can hold that
    # down when every other signal agrees.
    {"id": "date_prox",   "cluster": "temporal",   "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.55, "kind": "date_prox"},
    {"id": "po_ref",      "cluster": "reference",  "tier": 2, "weight": 2, "appl": 1.0, "cap": 0.80, "kind": "po_ref"},
    {"id": "currency",    "cluster": "commercial", "tier": 3, "weight": 1, "appl": 1.0, "cap": 0.90, "kind": "currency"},
]

_le.register_profile("invoice_duplicate", {
    "p0": 0.02, "alpha": 0.40, "floor": 0.55,
    "signals": _DUPLICATE_SIGNALS, "date_field": "invoice_date",
})

# The engine's own bands decide, rather than a threshold invented here. At/above auto_link
# the pair is a duplicate; between warn and auto it is raised for a human to confirm; below
# that nothing is said at all.
RAISE_BAND = _le._BAND_WARN      # 80.0 — the floor for saying anything
CERTAIN_BAND = _le._BAND_AUTO    # 92.0 — the floor for calling it critical


# ---------------------------------------------------------------------------
# Row shaping
# ---------------------------------------------------------------------------

def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _pence(amount: Any) -> Optional[int]:
    if amount is None:
        return None
    try:
        return int(round(float(amount) * 100))
    except (TypeError, ValueError):
        return None


def _norm_supplier(name: Any) -> str:
    if name is None:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split())


def _engine_row(inv: dict) -> dict:
    """The detector's row shape -> the row shape score_link's comparators read.

    supplier_id carries the normalised supplier (cmp_supplier compares resolved ids, and a
    normalised name IS the resolved identity here); converted_amount_usd carries the billed
    total, which is the comparable basis when both sides are invoices in one currency.
    """
    return {
        "invoice_id": inv.get("invoice_id"),
        "invoice_ref": inv.get("invoice_ref") or inv.get("invoice_id"),
        "invoice_date": inv.get("invoice_date"),
        "po_id": inv.get("po_id"),
        "supplier_id": _norm_supplier(inv.get("supplier_name")) or inv.get("supplier_id"),
        "converted_amount_usd": inv.get("total_amount"),
        "currency": inv.get("currency"),
        "country": inv.get("country"),
        "region": inv.get("region"),
    }


def score_pair(earlier: dict, later: dict) -> dict:
    """The relationship math for one candidate pair — score_link's full auditable result."""
    return _le.score_link(_engine_row(later), _engine_row(earlier), "invoice_duplicate",
                          later.get("lines") or [], earlier.get("lines") or [])


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def find_duplicates(invoices: list[dict], min_score: float = RAISE_BAND) -> list[dict]:
    """Score every plausible pair and return the ones the engine puts at or above
    ``min_score``. Each result is {'later', 'earlier', 'amount', 'score', 'band', 'link'} —
    the LATER invoice is flagged, because it is the one that should not have been paid, and
    ``amount`` is its full total (the whole payment is at risk, not a difference).

    Candidates are bucketed on (supplier, total to the penny) before scoring. Both are
    prerequisites of any duplicate — a different amount is a different bill — so bucketing
    changes no answer, and it keeps 12,408 live invoices from becoming 77M scored pairs.
    """
    buckets: dict[tuple[str, int], list[dict]] = {}
    for inv in invoices or []:
        supplier = _norm_supplier(inv.get("supplier_name")) or str(inv.get("supplier_id") or "")
        pence = _pence(inv.get("total_amount"))
        when = _as_date(inv.get("invoice_date"))
        # Absent data is not agreement. A non-positive total is a credit note or a nil
        # bill: nothing was paid twice.
        if not supplier or pence is None or pence <= 0 or when is None or not inv.get("invoice_id"):
            continue
        buckets.setdefault((supplier, pence), []).append(inv)

    out: list[dict] = []
    for rows in buckets.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (_as_date(r["invoice_date"]), str(r["invoice_id"])))
        for i, later in enumerate(rows):
            best = None
            for earlier in rows[:i]:
                if str(earlier["invoice_id"]) == str(later["invoice_id"]):
                    continue
                link = score_pair(earlier, later)
                if link["F"] < min_score:
                    continue
                if best is None or link["F"] > best[0]["F"]:
                    best = (link, earlier)
            if best is None:
                continue
            link, earlier = best
            # One finding per document, against the invoice it most strongly duplicates —
            # three identical invoices raise two findings, not three overlapping pairs.
            out.append({"later": later, "earlier": earlier,
                        "amount": round(float(later["total_amount"]), 2),
                        "score": link["F"], "band": link["decision"], "link": link})
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
           i.invoice_date,
           i.currency,
           i.country,
           i.region
      FROM proc.bp_invoice_trgt i
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
     WHERE i.invoice_id IS NOT NULL
"""

_LINES_SQL = """
    SELECT invoice_id, item_id, item_description, quantity, unit_price, line_amount
      FROM proc.bp_invoice_line_items_trgt
     WHERE invoice_id IS NOT NULL
"""


def load_invoices(cur) -> list[dict]:
    """The invoices the engine scores, each with its line items attached — the line set is
    one of the heaviest signals, so loading without it would quietly weaken every score.

    invoice_ref IS invoice_id here: bp_invoice_trgt has no separate reference column, and
    the id is the reference printed on the document.
    """
    cur.execute(_LOAD_SQL)
    rows = {}
    for invoice_id, po_id, supplier_name, total, when, currency, country, region in cur.fetchall():
        rows[invoice_id] = {"invoice_id": invoice_id, "po_id": po_id,
                            "supplier_name": supplier_name, "total_amount": total,
                            "invoice_date": when, "currency": currency,
                            "country": country, "region": region,
                            "invoice_ref": invoice_id, "lines": []}
    cur.execute(_LINES_SQL)
    for invoice_id, item_id, desc, qty, unit_price, amount in cur.fetchall():
        row = rows.get(invoice_id)
        if row is not None:
            row["lines"].append({"item_id": item_id, "item_description": desc,
                                 "quantity": qty, "unit_price": unit_price,
                                 "line_amount": amount})
    return list(rows.values())


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
            'duplicate_invoice', %s, %s, %s, false, 'open', %s)
"""


def _supporting(link: dict) -> str:
    """The signals that carried the score, named — so the note says what was compared and
    not merely that a number came out high."""
    label = {"ref_prox": "reference", "supplier_id": "supplier", "amount": "amount",
             "line_set": "line items", "date_prox": "date", "po_ref": "purchase order",
             "currency": "currency"}
    return ", ".join(label.get(s["id"], s["id"]) for s in link.get("signals", [])
                     if s.get("status") == "OK") or "no single signal"


def _note(dup: dict) -> str:
    earlier, link = dup["earlier"], dup["link"]
    when = _as_date(earlier.get("invoice_date"))
    return (f"possible duplicate of {earlier['invoice_id']}"
            f"{f' ({when:%Y-%m-%d})' if when else ''}: relationship score "
            f"{dup['score']:.1f}/100 ({dup['band']}), agreeing on {_supporting(link)}")


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
        doc_pk = str(dup["later"]["invoice_id"])
        if _already_raised(cur, doc_pk):
            continue
        raw_id, source_file = _source_of(cur, doc_pk)
        # A pair the engine is certain about is critical; one it puts in the warning band
        # is raised for a human to confirm, not asserted.
        severity = "critical" if dup["score"] >= CERTAIN_BAND else "warning"
        cur.execute(_INSERT_SQL, (
            raw_id, source_file, doc_pk, severity,
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
