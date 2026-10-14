"""A re-issued PO must keep its own identity, in the same convention the gateway parses."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services.extraction.po_revision import canonical_po_revision, normalise_approval, po_base


@pytest.mark.parametrize("extracted, text, expected", [
    ("4500018832", "Purchase Order 4500018832 Rev 3\nDate 2026-06-02", ("4500018832 (Rev 3)", 3)),
    ("4500018832", "PO No: 4500018832  Revision: 2", ("4500018832 (Rev 2)", 2)),
    ("PO-2025-0270", "PO-2025-0270 (Change Order 2)", ("PO-2025-0270 (Rev 2)", 2)),
    ("PO000101", "PO000101 - Amendment No. 4", ("PO000101 (Rev 4)", 4)),
    ("PO000101", "PO000101 Rev 1", ("PO000101", 1)),            # revision 1 stays unsuffixed
    ("PO000101", "PO000101 Rev 0 (original issue)", ("PO000101", 0)),
    ("PO000101", "Purchase order PO000101\nTotal 1,200", ("PO000101", None)),  # no marker: unchanged
    ("PO000101 (Rev 3)", "PO000101 Revision 3", ("PO000101 (Rev 3)", 3)),     # already canonical
])
def test_canonical_po_revision(extracted, text, expected):
    assert canonical_po_revision(extracted, text) == expected


def test_reads_the_marker_next_to_this_po_not_another():
    text = "Supersedes PO000099 Rev 7.\nPurchase Order PO000101\nTotal 1,200"
    assert canonical_po_revision("PO000101", text) == ("PO000101", None)


def test_po_base_strips_only_the_revision():
    assert po_base("PO-2025-0270 (Rev 2)") == "PO-2025-0270"
    assert po_base("PO-2025-0270") == "PO-2025-0270"


@pytest.mark.parametrize("raw, expected", [
    ("Approved", "approved"), ("Authorised by J. Smith", "approved"), ("Released", "approved"),
    ("Pending approval", "pending"), ("DRAFT", "pending"), ("Submitted for approval", "pending"),
    ("Not approved", "rejected"), ("Rejected", "rejected"),
    ("Cancelled", "cancelled"), ("VOID", "cancelled"),
    ("", None), (None, None), ("Net 30", None),
])
def test_normalise_approval(raw, expected):
    assert normalise_approval(raw) == expected


# The identifier repair in context_layer (_recover_identifiers) must not strip a
# revision back off: "the model included too much" is not what a revision suffix is.
# Without this, revision 3 of a PO was persisted under the bare number and overwrote
# revision 1 (found on a live run, 2026-09-25); quotes lost "(V2)" the same way.
from src.services.extraction.context_layer import _recover_identifiers  # noqa: E402


@pytest.mark.parametrize("field,value,text", [
    ("po_id", "PODM-77120 (Rev 3)", "PO Number: PODM-77120 Revision 3 PO Date: 12 August 2026"),
    ("quote_id", "QTE-2026-015 (V2)", "Quote Number: QTE-2026-015 (V2)\nDate 1 Jan"),
])
def test_identifier_repair_keeps_the_revision(field, value, text):
    assert _recover_identifiers({field: value}, text, valid_fields={field})[field] == value


def test_identifier_repair_still_strips_a_label():
    text = "PO Number: 1000587\nDate"
    out = _recover_identifiers({"po_id": "PO Number: 1000587"}, text, valid_fields={"po_id"})
    assert out["po_id"] == "1000587"


def _approval_hit(text):
    from src.services.extraction.pattern_registry import PatternRegistry, clear_cache
    clear_cache()
    for cp in PatternRegistry("purchase_order").patterns_for("approval_status"):
        for m in cp.anchor_re.finditer(text):
            vm = cp.value_re.search(text[m.end():m.end() + cp.max_span_after_anchor_chars])
            if vm:
                return vm.group(1)
    return None


@pytest.mark.parametrize("text,want", [
    ("PO Number: 4500018832 Status: Pending approval Supplier: Northwind", "pending"),
    ("Approval Status: Not approved", "rejected"),
    ("PO Status: Approved\nBuyer", "approved"),
    ("Status: Cancelled", "cancelled"),
])
def test_the_printed_approval_is_read_and_normalised(text, want):
    assert normalise_approval(_approval_hit(text)) == want


def test_a_status_that_is_not_an_approval_is_not_read():
    # "Status: Delivered" / a following label must never be taken as an approval.
    assert _approval_hit("Status: Delivered Supplier: Approved Vendors Ltd") is None
