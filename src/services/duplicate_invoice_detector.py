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
from datetime import date, datetime, timezone
from typing import Any, Optional

from src.services import linking_engine as _le
from src.services.formulas import ensure_registered, evaluate_many
# One FX implementation, shared with the reader: the same rates, the same GBP, the same
# refusal to convert what it cannot.
from src.services.value_summary_service import _get_rates, _to_gbp

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

    # Every candidate pair across every bucket, scored in ONE batch. The pairing
    # is unchanged; what changes is that the sweep writes a single audit record
    # rather than one per pair.
    pairs: list[tuple[dict, dict]] = []
    for rows in buckets.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: (_as_date(r["invoice_date"]), str(r["invoice_id"])))
        for i, later in enumerate(rows):
            for earlier in rows[:i]:
                if str(earlier["invoice_id"]) == str(later["invoice_id"]):
                    continue
                pairs.append((earlier, later))

    if not pairs:
        return []

    ensure_registered()
    batch = evaluate_many(
        "duplicate_invoice.pair_score",
        [{"earlier": e, "later": l} for e, l in pairs],
    )

    best_by_later: dict[str, tuple[dict, dict, dict]] = {}
    for (earlier, later), result in zip(pairs, batch):
        if result.unassessed:
            # Unscoreable is not "cleared". Say so rather than letting a refused
            # contract read as a clean bill of health for the later invoice.
            logger.warning(
                "duplicate scoring unassessed for %s vs %s: %s",
                later.get("invoice_id"), earlier.get("invoice_id"), result.why(),
            )
            continue
        link = result.value
        if link["F"] < min_score:
            continue
        key = str(later["invoice_id"])
        current = best_by_later.get(key)
        if current is None or link["F"] > current[0]["F"]:
            best_by_later[key] = (link, earlier, later)

    # One finding per document, against the invoice it most strongly duplicates —
    # three identical invoices raise two findings, not three overlapping pairs.
    return [
        {"later": later, "earlier": earlier,
         "amount": round(float(later["total_amount"]), 2),
         "score": link["F"], "band": link["decision"], "link": link}
        for link, earlier, later in best_by_later.values()
    ]


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

# invoice_total_incl_tax is what was actually billed and would actually be paid twice;
# invoice_amount (net) is the fallback for rows where the gross was never extracted.
_LOAD_SQL = """
    SELECT i.invoice_id,
           i.po_id,
           i.supplier_id,
           COALESCE(s.supplier_name, i.supplier_id) AS supplier_name,
           COALESCE(i.invoice_total_incl_tax, i.invoice_amount) AS total_amount,
           i.invoice_date,
           i.currency,
           i.country,
           i.region,
           i.invoice_status,
           i.invoice_paid_date,
           NULLIF(i.deal_id, '') AS deal_id
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
    for (invoice_id, po_id, supplier_id, supplier_name, total, when, currency,
         country, region, status, paid_date, deal_id) in cur.fetchall():
        rows[invoice_id] = {"invoice_id": invoice_id, "po_id": po_id,
                            "supplier_id": supplier_id,
                            "supplier_name": supplier_name, "total_amount": total,
                            "invoice_date": when, "currency": currency,
                            "country": country, "region": region,
                            "invoice_status": status, "invoice_paid_date": paid_date,
                            "deal_id": deal_id,
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


DETECTOR_TYPE = "Duplicate Invoice Recovery"


def payment_evidence(inv: dict) -> tuple[bool, str]:
    """What the corpus actually knows about whether this invoice was paid.

    Returns (paid_is_known, phrase). It matters because the two cases call for opposite
    actions: an unpaid duplicate is a payment to STOP, a paid one is money to GET BACK. The
    whole live corpus has invoice_status and invoice_paid_date NULL on all 12,408 rows, so
    the honest phrase is "if it was paid" — the opportunity is real either way (the money is
    at risk either way), but we never assert a payment we cannot see.
    """
    paid_date = _as_date(inv.get("invoice_paid_date"))
    status = str(inv.get("invoice_status") or "").strip().lower()
    if paid_date is not None:
        return True, f"paid on {paid_date:%Y-%m-%d} — recover from the supplier"
    if status in ("paid", "settled"):
        return True, f"marked {status} — recover from the supplier"
    if status:
        return False, f"status {status} — stop the payment if it has not gone out"
    return False, ("payment status not recorded on either invoice — recover if it was paid, "
                   "stop the payment if it has not gone out")


def _opportunity_record(dup: dict, rates: Optional[dict] = None) -> dict:
    """The duplicate as something to ACT on.

    Finding the duplicate and recovering the money are two different jobs. The discrepancy
    is the finding — it belongs to the document and lives in Data Validation & Actions. This
    is the recovery: it goes on the Opportunities pipeline with a financial impact and a
    stage, so somebody chases it and realised_savings_gbp records what actually came back.

    Anchored to the duplicate invoice via invoice_id, which is the SAME document the
    discrepancy names. value_summary_service.dedupe() keys on (deal_id, doc_pk), so the two
    collapse to one entry and the money is counted once — as verified (we found it), with
    the opportunity carrying the recovery rather than a second, potential-tier figure.
    """
    later, earlier, link = dup["later"], dup["earlier"], dup["link"]
    paid_known, phrase = payment_evidence(later)
    when = _as_date(earlier.get("invoice_date"))
    # financial_impact_gbp means GBP. The corpus bills in INR, USD, AED, GBP and EUR, so the
    # native total has to be converted before it goes in that column — stamping EUR 147,783
    # there renders as "£147.8K", a number nobody was ever billed. Unconvertible (no rate,
    # unknown currency) means NULL, never the native figure in disguise: the opportunity
    # still exists to be worked, it just has no GBP claim attached.
    currency = (later.get("currency") or "").upper() or None
    impact_gbp, _from = _to_gbp(dup["amount"], currency, rates)
    money = (f"{impact_gbp:,.2f} GBP" if impact_gbp is not None
             else f"{dup['amount']:,.2f} {currency or 'unknown currency'}")
    return {
        "opportunity_id": f"dupinv:{later['invoice_id']}",
        "opportunity_ref_id": f"duplicate_invoice_{earlier['invoice_id']}_{later['invoice_id']}",
        "detector_type": DETECTOR_TYPE,
        "supplier_id": later.get("supplier_id"),
        "supplier_name": later.get("supplier_name"),
        "item_id": str(later["invoice_id"]),
        "item_description": (
            f"Recover {money} — {later['invoice_id']} duplicates "
            f"{earlier['invoice_id']}{f' ({when:%Y-%m-%d})' if when else ''}; {phrase}"
        ),
        "financial_impact_gbp": impact_gbp,
        "invoice_id": str(later["invoice_id"]),
        "po_id": later.get("po_id"),
        "deal_id": later.get("deal_id"),
        "detected_on": datetime.now(timezone.utc),
        "source_records": [str(earlier["invoice_id"]), str(later["invoice_id"])]
                          + ([str(later["po_id"])] if later.get("po_id") else []),
        "calculation_details": {
            "duplicate_of": str(earlier["invoice_id"]),
            "relationship_score": dup["score"],
            "band": dup["band"],
            "signals": {s["id"]: s["status"] for s in link.get("signals", [])},
            "amount_native": dup["amount"],
            "currency": currency,
            "amount_gbp": impact_gbp,
            "payment_confirmed": paid_known,
            "payment_note": phrase,
        },
    }


def run_detector(conn=None) -> int:
    """Find duplicates and record the ones not already recorded. Returns rows written.

    Each duplicate produces two things, because finding money and getting it back are two
    different jobs: a discrepancy on the document (the finding) and an opportunity on the
    pipeline (the recovery). They share the invoice as their key so the money counts once.

    Idempotent: a document that already carries a duplicate_invoice finding is skipped, so
    the scheduler can call this on every promotion event without stacking findings. The
    opportunity upsert is keyed on the pair's content, so re-running never duplicates it
    and never demotes a recovery someone has already progressed.
    """
    if conn is not None:
        return _run(conn)
    from src.services.extraction.persistence import get_conn
    with get_conn() as own:
        return _run(own)


def _run(conn) -> int:
    from src.services.opportunity_store import upsert_opportunity

    cur = conn.cursor()
    dups = find_duplicates(load_invoices(cur))
    rates = _get_rates()
    written = opportunities = 0
    for dup in dups:
        doc_pk = str(dup["later"]["invoice_id"])
        # The opportunity is upserted even when the finding already exists: it is keyed on
        # the pair and never demotes a progressed stage, so this simply keeps the recovery
        # in step with the evidence.
        upsert_opportunity(cur, _opportunity_record(dup, rates))
        opportunities += 1
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
    logger.info("duplicate-invoice detector: %s candidate(s), %s new finding(s), "
                "%s recovery opportunit(y/ies)", len(dups), written, opportunities)
    return written
