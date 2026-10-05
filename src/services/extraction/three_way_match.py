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

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

from src.services.extraction.two_way_match import assign_lines

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

#: Why a line could not be checked. Each is a DENOMINATOR, never a failure.
NO_RECEIPT = "NO_RECEIPT"
UNVERIFIABLE_BY_RECEIPT = "UNVERIFIABLE_BY_RECEIPT"
UNVERIFIABLE_UOM = "UNVERIFIABLE_UOM"

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


def _units_agree(po_line: dict, counted: Iterable[dict]) -> bool:
    """Do the receipt and invoice lines count in the order's unit?

    A DIFFERENT unit refuses: `box` against `each` is the likeliest real
    failure, nothing here converts, and reporting a shortfall that is really a
    unit difference would be worse than silence. A MISSING unit does not
    refuse -- a delivery note often prints no unit at all, and absence is not
    disagreement. That is the one place this trusts the order's unit, and it is
    a deliberate choice between two imperfect readings, not an oversight.
    """
    po_unit = _norm_uom(po_line.get("unit_of_measure"))
    for line in counted:
        unit = _norm_uom(line.get("unit_of_measure"))
        if unit is not None and po_unit is not None and unit != po_unit:
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


def check(*, po_lines: list[dict], receipt_lines: list[dict],
          invoice_lines: list[dict], limits: Optional[dict] = None,
          po_id: Any = None) -> MatchResult:
    """Compare ordered / received / billed per PO line, over the whole set.

    Per line and over the set, not per document: two invoices that each pass
    alone can together bill more than arrived, and a line delivered in two
    parts has not been short-delivered twice.
    """
    limits = governed_tolerances() if limits is None else limits
    po_lines = list(po_lines or [])
    rec_by_idx = assign_receipt_lines(receipt_lines, po_lines, po_id=po_id)
    inv_by_idx = assign_lines(list(invoice_lines or []), po_lines, po_id)

    findings: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    assessed: list[Any] = []

    for pos, po_line in enumerate(po_lines):
        key = _line_key(po_line)
        if receipt_basis(po_line.get("unit_of_measure")) != "goods_receipt":
            unverifiable.append({"po_line": key, "reason": UNVERIFIABLE_BY_RECEIPT})
            continue

        receipts = [receipt_lines[i] for i, assigned in rec_by_idx.items()
                    if assigned is po_line]
        invoices = [invoice_lines[i] for i, assigned in inv_by_idx.items()
                    if assigned is po_line]

        if not _units_agree(po_line, [*receipts, *invoices]):
            unverifiable.append({"po_line": key, "reason": UNVERIFIABLE_UOM})
            continue

        billed = sum(_f(line.get("quantity")) or 0.0 for line in invoices)
        if not receipts and billed <= 0:
            # Nothing arrived and nothing was billed. Not a failure -- nobody
            # has claimed anything yet. Review Focus #2.
            unverifiable.append({"po_line": key, "reason": NO_RECEIPT})
            continue

        ordered = _f(po_line.get("quantity")) or 0.0
        # quantity_rejected is SUBTRACTED, never summed in: goods delivered and
        # refused were not received. Review Focus #5.
        received = sum(
            (_f(line.get("quantity_received")) or 0.0)
            - (_f(line.get("quantity_rejected")) or 0.0)
            for line in receipts
        )
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

    return MatchResult(findings, unverifiable, assessed)


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
    key = str(doc_pk).strip() if doc_pk is not None else ""
    kept = [r for r in persisted if str(r.get(id_field) or "").strip() != key] if key \
        else list(persisted)
    return kept + [{**line, id_field: doc_pk} for line in (own_lines or [])]


def check_against_receipts(doc_type: str, columns: dict,
                           line_items: list[dict]) -> list[Any]:
    """Findings from comparing this document's order against what arrived.

    Runs for an INVOICE (did we bill for more than arrived?) and for a GOODS
    RECEIPT (a note arriving after the invoice is exactly when an over-billing
    becomes visible, or stops being one).

    Returns `two_way_match.Discrepancy` rows. None of them blocks promotion:
    the design raises findings and holds nothing (§12).
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

    if doc_type == "invoice":
        invoices = _with_this_document(
            invoices, line_items, id_field="invoice_id",
            doc_pk=columns.get("invoice_id"))
    else:
        receipts = _with_this_document(
            receipts, line_items, id_field="grn_id", doc_pk=columns.get("grn_id"))

    if not receipts:
        # Nothing has been received against this order. That is not a finding --
        # the paperwork may simply not have arrived -- and saying so for every
        # invoice on a corpus with no receipts would bury the queue. Review
        # Focus #2, at the seam rather than only inside check().
        return []

    result = check(po_lines=po_lines, receipt_lines=receipts,
                   invoice_lines=invoices, po_id=po_id)

    out = []
    for f in result.findings:
        out.append(Discrepancy(
            field_name=f"po_line[{f['po_line']}]",
            issue_type=_ISSUE_TYPE[f["type"]],
            severity=_SEVERITY[f["type"]],
            blocks_promotion=False,
            raw_value=f"billed {f['billed']:g}",
            expected_value=f"received {f['received']:g}",
            computed_value=f"ordered {f['ordered']:g}",
            notes=_explain(f),
        ))
    return out


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
