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

from typing import Any

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
