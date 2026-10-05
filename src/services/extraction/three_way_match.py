"""The real three-way match: purchase order, goods receipt, invoice -- on quantity.

`two_way_match` compares an invoice to its order by VALUE. This compares three
documents by QUANTITY, which is the only comparison that can prove delivery: a
value reconciliation says the invoice agrees with the order's arithmetic, not
that anything arrived.

The PO line is the spine. Invoice lines and receipt lines both assign to it
through the same line matcher -- `two_way_match.assign_lines`, by delegation and
not a second copy -- so the two sides of the comparison can never disagree about
which ordered line they mean. The sums are then compared per PO line over the
WHOLE document set, because two invoices that each pass alone can together bill
more than was received; that is the lesson
`two_way_match._check_po_consumed_as_a_set` already encodes and this must not
have to relearn.

Nothing here blocks anything. It raises findings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

from src.services.extraction.two_way_match import assign_lines

log = logging.getLogger(__name__)

#: Receipt-line assignment is recorded under its own profile id so a stored
#: result can be told apart from the invoice side's. It is a version
#: identifier, not a description -- see the note on
#: `profile_registry_version` in two_way_match.
RECEIPT_LINE_PROFILE = "receipt_line_po_line"


def assign_receipt_lines(receipt_lines: list[dict], po_lines: list[dict],
                         *, po_id: Any) -> dict[int, dict]:
    """Receipt lines onto PO lines, by the same machinery the invoice uses.

    Delegation, not a parallel implementation: a second line-matcher would
    drift from the first and the two sides of the comparison would stop
    agreeing about which PO line they mean.

    The one adaptation is naming. A receipt counts in `quantity_received`
    while the matcher reads `quantity`, so the rows are normalised on the way
    in. `item_description` is already the shared name.
    """
    normalised = [
        {**line, "quantity": line.get("quantity_received")}
        for line in (receipt_lines or [])
    ]
    return assign_lines(normalised, po_lines, po_id)


#: Invoice-line assignment reuses the invoice side's own profile, unchanged.
INVOICE_LINE_PROFILE = "invoice_line_po_line"

#: What the three findings mean, so the names are not the documentation:
#:   NOTHING_RECEIVED     something was billed against a line and NO receipt
#:                        reached it. Distinct from a shortfall: "received 0"
#:                        would read as "we got some of it".
#:   BILLED_NOT_RECEIVED  more was billed than arrived, over the whole set of
#:                        invoices against that line.
#:   OVER_DELIVERED       more arrived than was ordered. A warning about the
#:                        delivery, not an accusation about the bill.
BILLED_NOT_RECEIVED = "BILLED_NOT_RECEIVED"
NOTHING_RECEIVED = "NOTHING_RECEIVED"
OVER_DELIVERED = "OVER_DELIVERED"

#: Why a line could not be checked. Each is a DENOMINATOR, never a failure,
#: and each must reach a reader -- a refusal nobody can see is a pass.
#:   NO_RECEIPT              nothing arrived and nothing was billed
#:   UNVERIFIABLE_BY_RECEIPT the unit is not something a delivery note can prove
#:   UNVERIFIABLE_UOM        the units cannot be compared, or one side printed none
#:   UNVERIFIABLE_QUANTITY   a quantity on one side is absent or nonsensical
#:   RECEIPT_LINE_UNPLACED   a receipt line names nothing on the order (order-level)
NO_RECEIPT = "NO_RECEIPT"
UNVERIFIABLE_BY_RECEIPT = "UNVERIFIABLE_BY_RECEIPT"
UNVERIFIABLE_UOM = "UNVERIFIABLE_UOM"
UNVERIFIABLE_QUANTITY = "UNVERIFIABLE_QUANTITY"
RECEIPT_LINE_UNPLACED = "RECEIPT_LINE_UNPLACED"

_TOLERANCE_POLICY = "receipt_tolerances"
_TOLERANCE_RULES = ("over_delivery_pct", "billed_over_received_qty")


@dataclass(frozen=True)
class MatchResult:
    """What the match concluded, and what it could not conclude.

    `assessed` is the denominator that makes a rate honest: a PO line only
    enters it when there was something to compare. `unverifiable` is every line
    that could not be checked, with the reason -- a line nobody sent a GRN for
    has not failed, and 39.5% of lines are measured in units no delivery note
    can ever prove.
    """

    findings: list[dict[str, Any]]
    unverifiable: list[dict[str, Any]]
    assessed: list[Any]


def governed_tolerances() -> dict[str, Any]:
    """The tolerances, from proc.bp_policy, or raise.

    There is no code-side default. A default is a number nobody agreed to that
    silently decides whether a supplier is over-billing; `governed_limits.limit`
    raises for an absent rule and that refusal is the feature
    (project_governed_limits).
    """
    from src.services import governed_limits

    return {rule: governed_limits.limit(_TOLERANCE_POLICY, rule)
            for rule in _TOLERANCE_RULES}


def _f(value: Any) -> Optional[float]:
    """A number, or None. Never a silent zero: '' and None are ABSENT, and
    absence must not read as 'nothing arrived'."""
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _norm_uom(value: Any) -> Optional[str]:
    """A unit's comparable form. None when the document printed none."""
    text = re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())
    if not text:
        return None
    # 'boxes' and 'box' are the same unit; 'each' must not become 'eac'.
    if len(text) > 3 and text.endswith("es"):
        text = text[:-2]
    elif len(text) > 3 and text.endswith("s"):
        text = text[:-1]
    return text or None


@lru_cache(maxsize=1)
def _basis_by_unit() -> dict[str, str]:
    """`{normalised unit: receipt_basis}` from proc.bp_uom_canonical, aliases included.

    Read once per process. `reset_unit_cache()` clears it; a governance reload
    or a test that changes the table must call that.
    """
    from src.services.db import get_conn

    out: dict[str, str] = {}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT uom_code, aliases, receipt_basis FROM proc.bp_uom_canonical")
        for code, aliases, basis in cur.fetchall():
            if not basis:
                continue
            for spelling in [code, *(aliases or [])]:
                key = _norm_uom(spelling)
                if key:
                    out.setdefault(key, basis)
    return out


def reset_unit_cache() -> None:
    _basis_by_unit.cache_clear()


def receipt_basis(unit: Any) -> str:
    """Can a delivery note prove a line measured in this unit?

    A unit the canonical table has never seen returns 'none', so the line reads
    as UNVERIFIABLE rather than as received or missing. Under-claiming is the
    safe direction: it lands in the not-assessed denominator where somebody can
    see it, instead of becoming a pass or a failure nobody asked for.
    """
    key = _norm_uom(unit)
    if key is None:
        return "none"
    return _basis_by_unit().get(key, "none")


def _units_confirmed(po_line: dict, counted: Iterable[dict]) -> bool:
    """Does every counted line explicitly agree with the order's unit?

    `box` against `each` is the likeliest real practical failure, nothing here
    converts, and reporting a shortfall that is really a unit difference would
    be worse than silence -- so a DIFFERENT unit refuses.

    A MISSING unit also refuses, and that is a change of mind. The first
    version let it through on the reasoning that absence is not disagreement,
    which is true, and which is safe for staying SILENT and not safe for
    ACCUSING. Measured in review: a PO of 40 `each`, a note printing `5` with
    no unit (five boxes of eight) and an invoice for 40 produced a critical
    over-billing of 35. Trusting the order's unit to raise a finding is
    guessing, and the one thing section 5 of the design refuses to do is guess.
    """
    po_unit = _norm_uom(po_line.get("unit_of_measure"))
    if po_unit is None:
        return False
    for line in counted:
        if _norm_uom(line.get("unit_of_measure")) != po_unit:
            return False
    return True


def _line_key(po_line: dict) -> Any:
    """How a PO line is named in a finding. `line_number` is the column; the
    plan's fixtures used `line_no`, and real callers pass rows straight from
    proc.bp_po_line_items_*, so both are accepted."""
    for field in ("line_number", "line_no"):
        if po_line.get(field) is not None:
            return po_line[field]
    return None


def _finding(kind: str, po_line: dict, ordered: float, received: float,
             billed: float, *, receipts: list[dict], invoices: list[dict]) -> dict[str, Any]:
    return {
        "type": kind,
        "po_id": po_line.get("po_id"),
        "po_line": _line_key(po_line),
        "description": po_line.get("item_description"),
        "unit_of_measure": po_line.get("unit_of_measure"),
        "ordered": ordered,
        "received": received,
        "billed": billed,
        # Both sides' source documents, so a finding can be read back to the
        # paper it came from without re-running the match.
        "receipt_sources": sorted({str(r.get("grn_id")) for r in receipts
                                   if r.get("grn_id")}),
        "invoice_sources": sorted({str(i.get("invoice_id")) for i in invoices
                                   if i.get("invoice_id")}),
    }


def _assign_by_po_line_ref(receipt_lines: list[dict],
                           po_lines: list[dict]) -> dict[int, dict]:
    """Receipt lines that NAME a PO line, placed by that name.

    A delivery note printing "PO Line 2" is telling us which ordered line it
    delivered, and that is better evidence than a description similarity
    score. `po_line_ref` was extracted, stored and selected for a fortnight
    before anything read it; a note with a shortened description or a bare item
    code went unplaced and its PO line then reported NOTHING_RECEIVED.
    """
    by_ref: dict[str, dict] = {}
    for po_line in po_lines:
        key = str(_line_key(po_line) or "").strip()
        if key:
            by_ref.setdefault(key, po_line)
    out: dict[int, dict] = {}
    for idx, line in enumerate(receipt_lines or []):
        ref = str(line.get("po_line_ref") or "").strip()
        if ref and ref in by_ref:
            out[idx] = by_ref[ref]
    return out


def check(*, po_lines: list[dict], receipt_lines: list[dict],
          invoice_lines: list[dict], limits: Optional[dict] = None,
          po_id: Any = None,
          basis_lookup: Optional[Any] = None) -> MatchResult:
    """Compare ordered / received / billed per PO line, over the whole set.

    Per line and over the set, not per document: two invoices that each pass
    alone can together bill more than arrived, and a line delivered in two
    parts has not been short-delivered twice.

    Nothing here guesses. Every quantity that cannot be read, every unit that
    cannot be compared and every receipt line that names nothing on the order
    lands in `unverifiable` -- NEVER as a zero that reads as a shortfall, and
    never as a silence that reads as a pass. `assessed` counts only the lines
    where all three numbers were genuinely comparable, which is what makes any
    rate computed from it honest.

    `basis_lookup` exists so the arithmetic can be tested without the unit
    table; the default reads proc.bp_uom_canonical.
    """
    limits = governed_tolerances() if limits is None else limits
    basis = basis_lookup if basis_lookup is not None else receipt_basis
    po_lines = list(po_lines or [])
    receipt_lines = list(receipt_lines or [])
    invoice_lines = list(invoice_lines or [])

    # By NAME first, then by description over the whole set for the rest.
    rec_by_idx = dict(_assign_by_po_line_ref(receipt_lines, po_lines))
    unnamed = [l for i, l in enumerate(receipt_lines) if i not in rec_by_idx]
    if unnamed:
        offsets = [i for i in range(len(receipt_lines)) if i not in rec_by_idx]
        for local, assigned in assign_receipt_lines(unnamed, po_lines,
                                                    po_id=po_id).items():
            rec_by_idx[offsets[local]] = assigned
    inv_by_idx = assign_lines(invoice_lines, po_lines, po_id)

    findings: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    assessed: list[Any] = []

    # A receipt line that names nothing on the order is a quantity that EXISTS
    # and could not be placed. Reported once, at order level -- and while one
    # is outstanding no PO line on this order may be called NOTHING_RECEIVED,
    # because the delivery it is missing may be the line we could not place.
    unplaced = [l for i, l in enumerate(receipt_lines) if i not in rec_by_idx]
    if unplaced:
        unverifiable.append({"po_line": None, "reason": RECEIPT_LINE_UNPLACED})

    for po_line in po_lines:
        key = _line_key(po_line)
        if basis(po_line.get("unit_of_measure")) != "goods_receipt":
            unverifiable.append({"po_line": key, "reason": UNVERIFIABLE_BY_RECEIPT})
            continue

        receipts = [receipt_lines[i] for i, assigned in rec_by_idx.items()
                    if assigned is po_line]
        invoices = [invoice_lines[i] for i, assigned in inv_by_idx.items()
                    if assigned is po_line]

        billed = _total(invoices, "quantity")
        if not receipts and billed is not None and billed <= 0:
            # Nothing arrived and nothing was billed. Not a failure -- nobody
            # has claimed anything yet. Review Focus #2.
            unverifiable.append({"po_line": key, "reason": NO_RECEIPT})
            continue

        if not _units_confirmed(po_line, [*receipts, *invoices]):
            unverifiable.append({"po_line": key, "reason": UNVERIFIABLE_UOM})
            continue

        # quantity_rejected is SUBTRACTED, never summed in: goods delivered and
        # refused were not received. Review Focus #5.
        received = _received(receipts)
        if billed is None or received is None:
            # One side's numbers could not be read. Reading an absent quantity
            # as zero accuses a supplier who delivered everything, and reading
            # an absent BILLED quantity as zero passes a line nobody checked.
            unverifiable.append({"po_line": key, "reason": UNVERIFIABLE_QUANTITY})
            continue

        ordered = _f(po_line.get("quantity")) or 0.0
        assessed.append(key)

        if not receipts:
            findings.append(_finding(NOTHING_RECEIVED, po_line, ordered, 0.0, billed,
                                     receipts=receipts, invoices=invoices))
        elif billed > received + limits["billed_over_received_qty"]:
            findings.append(_finding(BILLED_NOT_RECEIVED, po_line, ordered, received,
                                     billed, receipts=receipts, invoices=invoices))

        if ordered > 0 and received > ordered * (1 + limits["over_delivery_pct"]):
            findings.append(_finding(OVER_DELIVERED, po_line, ordered, received, billed,
                                     receipts=receipts, invoices=invoices))

    if unplaced:
        # Stated after the fact so the suppression is visible in the result and
        # not only in this comment.
        findings = [f for f in findings if f["type"] != NOTHING_RECEIVED]

    return MatchResult(findings, unverifiable, assessed)


def _total(lines: list[dict], field: str) -> Optional[float]:
    """The sum of `field` over `lines`, or None if ANY line's value is unreadable.

    Not `sum(... or 0)`: a line whose quantity did not parse makes the total
    unknown, and an unknown total must refuse rather than understate.
    """
    total = 0.0
    for line in lines:
        value = _f(line.get(field))
        if value is None:
            return None
        total += value
    return total


def _received(receipts: list[dict]) -> Optional[float]:
    """What actually arrived and was accepted, or None if that cannot be read."""
    total = 0.0
    for line in receipts:
        got = _f(line.get("quantity_received"))
        if got is None:
            return None
        refused = _f(line.get("quantity_rejected")) or 0.0
        if refused > got:
            # "1 received, 9 rejected" is a document that does not make sense.
            # The old arithmetic produced "against -8 each received".
            return None
        total += got - refused
    return total


# ---------------------------------------------------------------------------
# The ingestion seam: findings into the existing discrepancy queue
# ---------------------------------------------------------------------------
#: finding type -> the snake_case issue_type the discrepancy queue and the
#: Action Centre use. The match's own vocabulary stays upper-case because it is
#: the comparison's answer; the queue's is the house convention.
_ISSUE_TYPE = {
    BILLED_NOT_RECEIVED: "billed_not_received",
    NOTHING_RECEIVED: "nothing_received",
    OVER_DELIVERED: "over_delivered",
}
_SEVERITY = {
    BILLED_NOT_RECEIVED: "critical",
    NOTHING_RECEIVED: "critical",
    # More arrived than was ordered. Worth a look, not an accusation.
    OVER_DELIVERED: "warning",
}


def _load_po_lines(cur, po_id: str) -> list[dict]:
    """PO lines WITH their unit of measure.

    `two_way_match._load_po` does not select unit_of_measure -- it compares
    value, which does not need one -- and this comparison cannot work without
    it. _trgt first, then _stg, so a revision still staged is not preferred
    over the promoted one for the same PO.
    """
    for table in ("proc.bp_po_line_items_trgt", "proc.bp_po_line_items_stg"):
        try:
            cur.execute(
                f"SELECT po_id, line_number, item_description, quantity, unit_of_measure "
                f"FROM {table} WHERE po_id = %s ORDER BY line_number", (po_id,))
            rows = cur.fetchall()
        except Exception:  # noqa: BLE001 - table may be absent in some envs
            # A missing table, a renamed column or a permission error is
            # otherwise indistinguishable from "this order has no lines", and
            # the whole control then returns nothing with no trace. See
            # project_governance_fail_open_layers: an outage must not look like
            # "nothing to report".
            log.warning("three-way match: could not read %s for PO %s",
                        table, po_id, exc_info=True)
            continue
        if rows:
            return [{"po_id": r[0], "line_number": r[1], "item_description": r[2],
                     "quantity": r[3], "unit_of_measure": r[4]} for r in rows]
    return []


def _load_receipt_lines(cur, po_id: str) -> list[dict]:
    """Every receipt line against this order, from _trgt and _stg, deduplicated
    by grn_id: a receipt that has promoted appears in both tiers."""
    seen: set[str] = set()
    out: list[dict] = []
    for table in ("proc.bp_goods_receipt_line_items_trgt",
                  "proc.bp_goods_receipt_line_items_stg"):
        try:
            cur.execute(
                f"SELECT grn_id, line_no, item_description, quantity_received, "
                f"quantity_rejected, unit_of_measure, po_line_ref "
                f"FROM {table} WHERE po_id = %s ORDER BY grn_id, line_no", (po_id,))
            rows = cur.fetchall()
        except Exception:  # noqa: BLE001
            log.warning("three-way match: could not read %s for PO %s",
                        table, po_id, exc_info=True)
            continue
        for r in rows:
            if r[0] in seen:
                continue
            out.append({"grn_id": r[0], "line_no": r[1], "item_description": r[2],
                        "quantity_received": r[3], "quantity_rejected": r[4],
                        "unit_of_measure": r[5], "po_line_ref": r[6]})
        seen.update(r[0] for r in rows if r[0])
    return out


def _load_invoice_lines(cur, po_id: str) -> list[dict]:
    """Every invoice line against this order, over the whole set.

    One invoice at a time cannot see two invoices that each fit alone and
    together bill more than arrived. Deduplicated by invoice_id for the same
    reason the receipts are.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for table in ("proc.bp_invoice_line_items_trgt", "proc.bp_invoice_line_items_stg"):
        try:
            cur.execute(
                f"SELECT invoice_id, line_no, item_description, quantity, unit_of_measure "
                f"FROM {table} WHERE po_id = %s ORDER BY invoice_id, line_no", (po_id,))
            rows = cur.fetchall()
        except Exception:  # noqa: BLE001
            log.warning("three-way match: could not read %s for PO %s",
                        table, po_id, exc_info=True)
            continue
        for r in rows:
            if r[0] in seen:
                continue
            out.append({"invoice_id": r[0], "line_no": r[1], "item_description": r[2],
                        "quantity": r[3], "unit_of_measure": r[4]})
        seen.update(r[0] for r in rows if r[0])
    return out


def _with_this_document(persisted: list[dict], own_lines: list[dict],
                        *, id_field: str, doc_pk: Any) -> list[dict]:
    """The persisted set with THIS document's lines substituted in.

    Two reasons it is a substitution and not an append. The document being
    extracted right now is not in `_stg` yet, so without this its own lines are
    invisible to the set view. And on a RE-READ the old version IS in `_stg`,
    so appending would count the same invoice twice and manufacture an
    over-billing -- the exact failure
    project_reread_does_not_refresh_trgt records for the value path.
    """
    def norm(value: Any) -> str:
        # Case- and separator-insensitive, like the PO side already is.
        # "INV-1" and "inv-1" are one invoice; comparing them with a plain
        # strip() kept BOTH rows, counted the document's lines twice and
        # manufactured an over-billing -- the exact failure this function
        # exists to prevent. Measured in review.
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    key = norm(doc_pk)
    kept = [r for r in persisted if norm(r.get(id_field)) != key] if key \
        else list(persisted)
    return kept + [{**line, id_field: doc_pk} for line in (own_lines or [])]


#: Reason -> the snake_case issue_type a refusal is filed under. Informational
#: and never blocking: these are the DENOMINATOR, not failures. They exist so
#: section 6 of the design is honoured -- "the system must say which lines it
#: could not check" -- at a surface a person reads, rather than only inside a
#: return value.
_REFUSAL_ISSUE_TYPE = {
    UNVERIFIABLE_BY_RECEIPT: "line_not_receivable",
    UNVERIFIABLE_UOM: "receipt_unit_not_comparable",
    UNVERIFIABLE_QUANTITY: "receipt_quantity_unreadable",
    RECEIPT_LINE_UNPLACED: "receipt_line_not_on_po",
}
#: NO_RECEIPT is deliberately absent. Every PO line on a corpus with no
#: delivery notes is in that state, and filing one row per line would bury the
#: queue under paperwork nobody has sent yet. It is visible as `lines_assessed`
#: being zero on the deal instead.

_REFUSAL_NOTE = {
    UNVERIFIABLE_BY_RECEIPT:
        "a line measured in this unit cannot be proved by a delivery note, so "
        "what was billed against it has not been compared with what arrived",
    UNVERIFIABLE_UOM:
        "the delivery note and the order do not count this line in the same "
        "unit, or one of them printed no unit, so the quantities cannot be "
        "compared and nothing is assumed",
    UNVERIFIABLE_QUANTITY:
        "a quantity on one side of this line could not be read, so what was "
        "billed has not been compared with what arrived",
    RECEIPT_LINE_UNPLACED:
        "a line on the delivery note names nothing on this order, so the "
        "delivery it records cannot be matched to an ordered line",
}


def check_against_receipts(doc_type: str, columns: dict,
                           line_items: list[dict]) -> list[Any]:
    """Compare this document's order against what arrived, and RECORD the answer.

    Runs for an INVOICE (did we bill for more than arrived?) and for a GOODS
    RECEIPT (a note arriving after the invoice is exactly when an over-billing
    becomes visible, or stops being one).

    Returns `two_way_match.Discrepancy` rows for the CURRENT document only.
    Everything else it has to say -- a gap belonging to another invoice, a
    refusal, a gap that has now closed -- it writes itself, because the caller
    can only file against the document it is reading. None of it blocks: the
    design raises findings and holds nothing.
    """
    from src.services.db import get_conn
    from src.services.extraction.two_way_match import Discrepancy
    from src.services.linking_engine import _pick_po

    if doc_type not in ("invoice", "goods_receipt"):
        return []
    cited = (columns.get("po_id") or "")
    if not str(cited).strip():
        return []

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            po_row = _pick_po(cur, str(cited).strip(), cols="t.po_id")
        except Exception:  # noqa: BLE001
            log.warning("three-way match: could not resolve PO %r", cited, exc_info=True)
            return []
        if not po_row:
            # An unknown PO is two_way_match's finding to raise, not a second one.
            return []
        po_id = po_row["po_id"]
        po_lines = _load_po_lines(cur, po_id)
        if not po_lines:
            return []
        receipts = _load_receipt_lines(cur, po_id)
        invoices = _load_invoice_lines(cur, po_id)

    own_pk = columns.get("invoice_id") if doc_type == "invoice" else columns.get("grn_id")
    if doc_type == "invoice":
        invoices = _with_this_document(
            invoices, line_items, id_field="invoice_id", doc_pk=own_pk)
    else:
        receipts = _with_this_document(
            receipts, line_items, id_field="grn_id", doc_pk=own_pk)

    if not receipts:
        # Nothing has been received against this order. That is not a finding --
        # the paperwork may simply not have arrived -- and saying so for every
        # invoice on a corpus with no receipts would bury the queue. Review
        # Focus #2, at the seam rather than only inside check().
        return []

    result = check(po_lines=po_lines, receipt_lines=receipts,
                   invoice_lines=invoices, po_id=po_id)

    # The receipt carries the match's denominator, so the deal overview can tell
    # "checked and clean" from "nothing was checkable". Written for whichever
    # receipts are on the order, because the assessment is of the SET.
    _record_outcome(receipts, result)

    out: list[Any] = []
    foreign: list[tuple[str, dict]] = []
    for f in result.findings:
        # A gap is contained in the INVOICE that over-billed, not in whichever
        # document happened to be read when it was spotted (design section 8).
        # Filing it against the note produced two rows for one gap -- one on the
        # note, one on the invoice after a re-read -- each needing separate
        # resolution, with the deal staying false until both were cleared.
        owners = f.get("invoice_sources") or []
        for invoice_id in owners:
            if doc_type == "invoice" and str(invoice_id) == str(own_pk):
                out.append(_as_discrepancy(Discrepancy, f))
            else:
                foreign.append((str(invoice_id), f))
        if not owners:
            out.append(_as_discrepancy(Discrepancy, f))

    for reason_row in result.unverifiable:
        reason = reason_row["reason"]
        if reason not in _REFUSAL_ISSUE_TYPE:
            continue
        out.append(Discrepancy(
            field_name=(f"po_line[{reason_row['po_line']}]"
                        if reason_row.get("po_line") is not None else "receipt_lines"),
            issue_type=_REFUSAL_ISSUE_TYPE[reason],
            severity="info",
            blocks_promotion=False,
            raw_value=str(po_id),
            notes=f"purchase order {po_id}: {_REFUSAL_NOTE[reason]}",
        ))

    _write_foreign_findings(foreign, po_id=po_id)
    _clear_closed_gaps(po_id, result, own_doc_type=doc_type, own_pk=own_pk)
    return out


def _as_discrepancy(Discrepancy, f: dict):
    return Discrepancy(
        field_name=f"po_line[{f['po_line']}]",
        issue_type=_ISSUE_TYPE[f["type"]],
        severity=_SEVERITY[f["type"]],
        blocks_promotion=False,
        raw_value=f"billed {f['billed']:g}",
        expected_value=f"received {f['received']:g}",
        computed_value=f"ordered {f['ordered']:g}",
        notes=_explain(f),
    )


def _record_outcome(receipts: list[dict], result: "MatchResult") -> None:
    """Write `lines_assessed` / `lines_unverifiable` onto every receipt on the order.

    Best effort: a bookkeeping write must never lose an extraction. It is
    logged rather than swallowed, because an outage here makes the deal read
    "not assessed", and silence about that is how a fail-open layer hides.
    """
    from src.services.db import get_conn

    grns = sorted({str(r.get("grn_id")) for r in receipts if r.get("grn_id")})
    if not grns:
        return
    assessed, refused = len(result.assessed), len(result.unverifiable)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            for table in ("proc.bp_goods_receipt_stg", "proc.bp_goods_receipt_trgt"):
                cur.execute(
                    f"UPDATE {table} SET lines_assessed = %s, lines_unverifiable = %s "
                    f"WHERE grn_id = ANY(%s)", (assessed, refused, grns))
    except Exception:  # noqa: BLE001
        log.warning("three-way match: could not record the outcome for %s "
                    "(the deal will read NOT ASSESSED)", grns, exc_info=True)


def _write_foreign_findings(foreign: list[tuple[str, dict]], *, po_id: Any) -> None:
    """File a gap against an invoice that is not the document being read.

    Two invoices can together bill more than arrived; the gap belongs to both,
    and neither may be the document in hand. The caller can only file against
    what it is reading, so these are written here, through the ordinary
    discrepancy writer, so the open-row key and the source_file normalisation
    are the same ones every other finding gets.
    """
    if not foreign:
        return
    from src.services.db import get_conn
    from src.services.extraction import persistence

    try:
        with get_conn() as conn, conn.cursor() as cur:
            for invoice_id, f in foreign:
                cur.execute(
                    "SELECT source_file FROM proc.bp_invoice_raw "
                    "WHERE doc_pk_candidate = %s AND source_file IS NOT NULL "
                    "ORDER BY raw_id DESC LIMIT 1", (invoice_id,))
                row = cur.fetchone()
                # source_file is part of the open-row key and is NOT NULL on the
                # table (project_findings_key_per_document). An invoice loaded
                # from outside the pipeline has no _raw row and so no path; a
                # deterministic per-document key keeps the row addressable and
                # keeps one invoice's gap from colliding with another's. It is
                # not a file path and is not pretending to be one.
                source_file = row[0] if row else f"po:{po_id}/invoice:{invoice_id}"
                persistence.write_discrepancies(
                    doc_type="invoice", raw_id=None, source_file=source_file,
                    doc_pk_candidate=invoice_id,
                    discrepancies=[_as_discrepancy(_Discrepancy(), f)],
                )
    except Exception:  # noqa: BLE001
        log.warning("three-way match: could not file findings against %s",
                    [i for i, _ in foreign], exc_info=True)


def _Discrepancy():
    from src.services.extraction.two_way_match import Discrepancy
    return Discrepancy


def _clear_closed_gaps(po_id: Any, result: "MatchResult", *,
                       own_doc_type: str, own_pk: Any) -> None:
    """Resolve the quantity findings that this run no longer raises.

    The module's whole claim for running on the receipt side is that a note
    arriving after the bill is the moment an over-billing stops being one. That
    was computed and never recorded: nothing in this codebase resolved an
    extraction discrepancy, so a closed gap stayed open in the Action Centre
    for ever and the deal stayed `false`. Resolved, not deleted -- the row is
    the record that it was once true.
    """
    from src.services.db import get_conn

    still_open = {
        (str(inv), f"po_line[{f['po_line']}]", _ISSUE_TYPE[f["type"]])
        for f in result.findings
        for inv in (f.get("invoice_sources") or [str(own_pk)])
    }
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT discrepancy_id, doc_type, doc_pk_candidate, field_name,
                          issue_type
                     FROM proc.bp_extraction_discrepancy
                    WHERE status = 'open' AND issue_type = ANY(%s)
                      AND raw_value LIKE 'billed %%'
                      AND (notes LIKE %s OR raw_value = %s)""",
                (list(_ISSUE_TYPE.values()), f"%purchase order {po_id} line%", str(po_id)))
            for did, dtype, pk, field, issue in cur.fetchall():
                if (str(pk), field, issue) in still_open:
                    continue
                if dtype == "goods_receipt" or (str(pk), field, issue) not in still_open:
                    cur.execute(
                        "UPDATE proc.bp_extraction_discrepancy "
                        "SET status = 'resolved', resolved_at = NOW(), "
                        "    resolved_by = 'three_way_match', "
                        "    resolution_outcome = 'accepted' "
                        "WHERE discrepancy_id = %s", (did,))
                    log.info("three-way match: gap %s on %s closed", did, pk)
    except Exception:  # noqa: BLE001
        log.warning("three-way match: could not clear closed gaps on PO %s",
                    po_id, exc_info=True)


def _explain(f: dict) -> str:
    """The finding in the words a buyer would use."""
    desc = str(f.get("description") or "this line")[:60]
    unit = f" {f['unit_of_measure']}" if f.get("unit_of_measure") else ""
    grns = ", ".join(f.get("receipt_sources") or []) or "none"
    invs = ", ".join(f.get("invoice_sources") or []) or "this invoice"
    if f["type"] == NOTHING_RECEIVED:
        return (f"'{desc}' on purchase order {f.get('po_id')} line {f.get('po_line')}: "
                f"{f['billed']:g}{unit} billed ({invs}) and no delivery recorded at all")
    if f["type"] == BILLED_NOT_RECEIVED:
        return (f"'{desc}' on purchase order {f.get('po_id')} line {f.get('po_line')}: "
                f"{f['billed']:g}{unit} billed ({invs}) against {f['received']:g}{unit} "
                f"received (receipts: {grns}) -- "
                f"{f['billed'] - f['received']:g}{unit} more than arrived")
    return (f"'{desc}' on purchase order {f.get('po_id')} line {f.get('po_line')}: "
            f"{f['received']:g}{unit} received against {f['ordered']:g}{unit} ordered "
            f"(receipts: {grns})")
