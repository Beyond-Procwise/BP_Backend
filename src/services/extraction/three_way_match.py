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

Over-billing hides in the aggregate, though, and comparing one document at a time cannot
see it. Two invoice lines that both land on the same PO line each pass a check against
that line's full authorised total while together billing twice it; two invoices each
citing the same PO at 60% of its value each pass the header check while together billing
120%. Both are now checked as a set: the lines are assigned to PO lines over the whole
document at once rather than one at a time, and what a PO line -- or a whole PO -- has
been billed in total is compared against what it authorised.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from src.services.extraction.persistence import Discrepancy, get_conn
from src.services.linking_engine import _norm_po, _PO_NORM_SQL
from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    resolve,
)

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


# ---------------------------------------------------------------------------
# Which PO line is each invoice line billing for?
# ---------------------------------------------------------------------------
# Scores that keep _match_po_line's precedence exactly: an exact key beats a
# containment, and a containment beats any token overlap however convincing.
_EXACT_SCORE = 1.0
_CONTAINED_SCORE = 0.9
_OVERLAP_CEILING = 0.8      # strictly below _CONTAINED_SCORE
_OVERLAP_FLOOR = 0.5        # the threshold _match_po_line has always used

# How many PO lines one invoice line is allowed to compete for. Every candidate
# pair is a variable, so keeping all of them makes the model grow with the
# product of the two line counts: a 30-line invoice against a 30-line purchase
# order was 900 edges and 0.58s, and a 200-line document would have run for
# minutes inside extraction. Keeping each line's best few is linear instead.
#
# It is a bound, and worth stating as one: an assignment could in principle want
# to push a line past its fourth choice, and this will not let it. That needs
# four other lines to outbid it on all four, which only happens in a document
# whose descriptions are near-identical — and there the fallback below matches
# it anyway. The best candidate is never dropped, so nothing this bound does can
# change a document whose lines do not compete.
_MAX_CANDIDATES_PER_LINE = 4

_LINE_PROFILE = "invoice_line_po_line"
# Prefer to give each invoice line a PO line of its own. Preferring, not
# requiring: see _assign_lines.
_LINE_RULE = CardinalityRule(_LINE_PROFILE, "1:1")


def _line_log_odds(score: float) -> float:
    """A match score as log-odds, for the resolver's objective. Monotone, so a
    document whose lines do not compete resolves to exactly what matching them
    one at a time would have returned."""
    p = min(max(score, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _candidate_po_lines(desc: Any, po_lines: list[dict]) -> list[tuple[int, float]]:
    """Every PO line this invoice line could be billing for, with a score.

    Eligibility is exactly what `_match_po_line` has always allowed — an exact
    key, one description containing the other, or a token overlap of at least
    0.5. The difference is that this returns all of them instead of the first
    or best, so the assignment below has something to choose between.
    """
    key = _norm_item(desc)
    if not key:
        return []
    words = set(key.split())
    out: list[tuple[int, float]] = []
    for i, line in enumerate(po_lines):
        pk = _norm_item(line.get("item_description"))
        if not pk:
            continue
        if pk == key:
            out.append((i, _EXACT_SCORE))
            continue
        if pk.startswith(key) or key.startswith(pk) or pk in key or key in pk:
            out.append((i, _CONTAINED_SCORE))
            continue
        pw = set(pk.split())
        if not pw or not words:
            continue
        overlap = len(words & pw) / len(words | pw)
        if overlap >= _OVERLAP_FLOOR:
            out.append((i, _OVERLAP_CEILING * overlap))
    # Best first, PO line order breaking ties, then bounded.
    out.sort(key=lambda c: (-c[1], c[0]))
    return out[:_MAX_CANDIDATES_PER_LINE]


def _assign_lines(line_items: list[dict], po_lines: list[dict],
                  po_id: Any) -> dict[int, dict]:
    """Which PO line each invoice line is billing for, decided over the whole
    document at once rather than one line at a time.

    Matching each line independently gives every line its own favourite, which
    goes wrong in two ways a set view fixes. Two invoice lines can settle on the
    same PO line while another PO line is then reported as never billed. And a
    line's favourite can be another line's only option: the first line takes it
    on a 0.6 overlap, the second needed it at 0.9, and both end up on the same
    PO line with a real one left over. Assigning the document as a set resolves
    both, and the tie-break is the resolution layer's documented rule rather
    than the order the PO lines happened to come out of the database in.

    **Preferring a PO line of one's own is not requiring one.** Split billing —
    two invoice lines against a single ordered item — is ordinary, so a line the
    one-to-one pass could not place falls back to its own best match rather than
    being left unplaced. This function never causes an accusation; a line it
    cannot match at all is one `_match_po_line` could not match either. What
    happens when two lines do share a PO line is a finding about the total
    billed against it, raised in `check_against_po`, not a refusal to match.
    """
    edges = []
    for idx, item in enumerate(line_items or []):
        for i, score in _candidate_po_lines(item.get("item_description"), po_lines):
            edges.append(CandidateEdge(
                source_id=f"line:{idx}", target_id=f"po_line:{i}",
                log_odds=_line_log_odds(score), confidence=score,
                profile_id=_LINE_PROFILE, consumes={},
            ))

    assigned: dict[int, dict] = {}
    if edges:
        result = resolve(ResolutionRequest(
            request_id=f"three_way_match:{po_id}",
            edges=tuple(edges), capacities=(), rules=(_LINE_RULE,),
            profile_registry_version="three_way_match/line_v1",
        ))
        for link in result.links:
            assigned[int(link.source_id.split(":", 1)[1])] = po_lines[
                int(link.target_id.split(":", 1)[1])
            ]

    for idx, item in enumerate(line_items or []):
        if idx in assigned:
            continue
        fallback = _match_po_line(item.get("item_description"), po_lines)
        if fallback is not None:
            assigned[idx] = fallback
    return assigned


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


# Document furniture that table extraction captures as "line items": the
# commercial-terms block, validity, and the confidentiality footer. Exact
# whole-description matches only (a real charge like "Payment processing fee"
# must never match), plus the two structural footer signatures.
_NON_CHARGE_DESCRIPTIONS = {
    "commercial terms", "payment terms", "terms", "term", "uplift", "payment",
    "price validity", "validity", "currency", "delivery", "date", "notes",
    "rfq reference", "quote ref", "quote reference",
}
_FOOTER_RE = re.compile(r"·.*·|commercial-in-confidence", re.IGNORECASE)


def is_non_charge_line(description: Any) -> bool:
    """True when a captured line is document furniture (terms block, footer),
    not a charge. Used to keep line-quality warnings and three-way-match
    findings honest; the extracted row itself is never deleted."""
    s = str(description or "").strip().lower()
    if not s:
        return False
    if s in _NON_CHARGE_DESCRIPTIONS:
        return True
    return bool(_FOOTER_RE.search(s))


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


def _billed_by_other_invoices(po_id: str, exclude_invoice_id: Any,
                             currency: Any) -> tuple[float, list[str], bool]:
    """What other invoices have already billed against this purchase order.

    Returns (total, invoice_ids, complete). ``complete`` is False when some
    invoice on the PO had to be left out because its currency differs from the
    PO's, or because the lookup failed — the total is then a floor rather than
    the whole story, and the caller says so.

    Net amounts, matching the header check above: it compares the document's
    `invoice_amount` against the PO's `total_amount`, so the siblings have to be
    added on the same basis or the comparison is between two different things.
    """
    canonical = _norm_po(po_id)
    if not canonical or not currency:
        return 0.0, [], False
    cond = _PO_NORM_SQL.format(col="po_id")
    exclude = str(exclude_invoice_id) if exclude_invoice_id else ""
    total, ids, complete = 0.0, [], True
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            # One row per invoice_id across both tiers: a document staged and then
            # promoted is one invoice, not two.
            cur.execute(
                "SELECT invoice_id, MAX(invoice_amount), MAX(currency) FROM ("
                f"  SELECT invoice_id, invoice_amount, currency FROM proc.bp_invoice_stg WHERE {cond} = %s"
                "  UNION ALL"
                f"  SELECT invoice_id, invoice_amount, currency FROM proc.bp_invoice_trgt WHERE {cond} = %s"
                ") x WHERE invoice_id IS NOT NULL AND invoice_id <> %s GROUP BY invoice_id",
                (canonical, canonical, exclude),
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — a sibling lookup must not lose the other findings
        log.exception("three-way match: could not read sibling invoices for PO %s", po_id)
        return 0.0, [], False

    want = str(currency).strip().upper()
    for invoice_id, amount, cur_code in rows:
        value = _f(amount)
        if value is None or not cur_code or str(cur_code).strip().upper() != want:
            complete = False
            continue
        total += value
        ids.append(str(invoice_id))
    return round(total, 2), sorted(ids), complete


def _check_po_consumed_as_a_set(po: dict, columns: dict,
                                doc_total: Optional[float]) -> list[Discrepancy]:
    """Do the invoices citing this PO, together, bill more than it authorised?"""
    po_total = _f(po.get("total_amount"))
    po_currency = po.get("currency")
    doc_currency = columns.get("currency")
    if not po_total or doc_total is None or not po_currency:
        return []
    if not doc_currency or str(doc_currency).strip().upper() != str(po_currency).strip().upper():
        return []

    others, ids, complete = _billed_by_other_invoices(
        po["po_id"], columns.get("invoice_id"), po_currency
    )
    if not ids:
        return []

    combined = round(others + doc_total, 2)
    if combined <= po_total or _agrees(combined, po_total):
        return []

    this_one = str(columns.get("invoice_id") or "this invoice")
    claimants = ", ".join(ids + [this_one])
    caveat = ("" if complete else
              " (invoices in another currency are excluded, so the real total is higher)")
    return [Discrepancy(
        field_name="invoice_amount",
        issue_type="po_over_consumed",
        severity="critical",
        blocks_promotion=False,
        raw_value=f"{combined:.2f}",
        expected_value=f"{po_total:.2f}",
        computed_value=f"+{round(combined - po_total, 2):.2f}",
        notes=(
            f"purchase order {po['po_id']} authorised {po_total:,.2f} {po_currency}; "
            f"{claimants} bill {combined:,.2f} combined, "
            f"{round(combined - po_total, 2):,.2f} more than was ordered{caveat}"
        ),
    )]


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
    # Assigned as a set, not one line at a time: see _assign_lines.
    assigned = _assign_lines(line_items or [], po_lines, po["po_id"])
    matched_po_lines: set[int] = set()
    claims: dict[int, list[tuple[Any, float]]] = {}

    for idx, li in enumerate(line_items or []):
        desc = li.get("item_description")
        key = _norm_item(desc)
        amt = _f(li.get("line_amount")) or _f(li.get("line_total")) or _f(li.get("total_amount"))
        if not key:
            continue

        po_line = assigned.get(idx)
        if po_line is not None:
            matched_po_lines.add(id(po_line))
            if amt is not None:
                claims.setdefault(id(po_line), []).append((desc, amt))
        if po_line is None:
            # "Charged but not on the PO" requires a charge: a row with no
            # money is furniture or an extraction gap (both surfaced
            # elsewhere), and terms/footer rows are never charges at all.
            if amt is None or is_non_charge_line(desc):
                continue
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

    # --- is one PO line being billed twice over? ------------------------------------
    # Each line above was compared against the PO line's FULL authorised total, which
    # every one of them can pass while together billing more than was ordered. Two
    # invoice lines at 60% of a PO line are 120% of it, and neither is individually
    # over-billing. Only the sum says so.
    for pl in po_lines:
        billed = claims.get(id(pl), [])
        if len(billed) < 2:
            continue
        po_amt = _f(pl.get("line_total"))
        total = round(sum(a for _, a in billed), 2)
        if not po_amt or total <= po_amt or _agrees(total, po_amt):
            continue
        out.append(Discrepancy(
            field_name="line_items",
            issue_type="po_line_over_consumed",
            severity="critical",
            blocks_promotion=False,
            raw_value=f"{total:.2f}",
            expected_value=f"{po_amt:.2f}",
            computed_value=f"+{round(total - po_amt, 2):.2f}",
            notes=(
                f"'{str(pl.get('item_description'))[:60]}' was authorised at "
                f"{po_amt:,.2f} on purchase order {po['po_id']} and is billed "
                f"{len(billed)} times on this document — "
                + ", ".join(f"{str(d)[:40]} {a:,.2f}" for d, a in billed)
                + f" — {round(total - po_amt, 2):,.2f} more than was ordered"
            ),
        ))

    # --- is the PO being billed twice over, across documents? -----------------------
    # The header check above asks whether THIS invoice exceeds the PO. Two invoices at
    # 60% of it each pass that and together bill 120%, so the question has to be asked
    # of the set. Only invoices stating the PO's own currency are added up: converting
    # here would mean inventing a rate, and a total nobody can verify is worse than no
    # finding at all.
    if doc_type == "invoice" and po_total:
        out.extend(_check_po_consumed_as_a_set(po, columns, doc_total))

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
