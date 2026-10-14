from decimal import Decimal as D

from src.services.triage.link import link
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def test_item_code_match_links_with_full_confidence():
    ds = deal(po(), inv())
    links = link(ds, CFG)
    (lk,) = links.line_links
    assert lk.po_line.line_ref == "1" and lk.confidence == 1.0
    assert links.invoice_po["INV-1"].doc_id == "PO-1"


def test_repeated_item_prefers_same_price_line():
    ds = deal(po(lines=[line(1, price="10.00"), line(2, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line.line_ref == "2"


def test_description_fallback_links_below_full_confidence():
    ds = deal(po(lines=[line(1, item=None, desc="Widget large")]),
              inv(lines=[line(1, item=None, desc="Widgets, large")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line is not None and 0.5 <= lk.confidence < 1.0


def test_dissimilar_line_is_unlinked():
    ds = deal(po(), inv(lines=[line(1, item="FRT", desc="Expedited freight")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line is None


def test_po_number_on_lines_is_used_when_header_is_blank():
    ds = deal(po(), inv(po_id=None, lines=[line(1, po_id="PO-1")]))
    assert link(ds, CFG).invoice_po["INV-1"].doc_id == "PO-1"


def test_bad_and_missing_po_references():
    ds = deal(po(), inv("INV-1", po_id="PO-404"), inv("INV-2", po_id=None))
    links = link(ds, CFG)
    assert links.bad_refs == {"INV-1"} and links.no_ref == {"INV-2"}
    assert links.line_links == []


def test_itemised_lines_roll_up_into_one_po_line():
    ds = deal(po(lines=[line(1), line(2, item=None, desc="Installation", qty="1", price="5000.00")]),
              inv(lines=[line(1)] + [line(n, item=None, desc=f"Day {n}", qty="1", price="1250.00")
                                     for n in (2, 3, 4, 5)]))
    links = link(ds, CFG)
    rolled = [lk for lk in links.line_links if lk.rollup]
    assert len(rolled) == 4 and {lk.po_line.line_ref for lk in rolled} == {"2"}


def test_po_to_quote():
    ds = deal(quote(), po(quote_ref="Q-1"), inv())
    assert link(ds, CFG).po_quote["PO-1"].doc_id == "Q-1"


# PO revisions (build 4): an invoice prints the bare PO number, never "(Rev 3)", and must
# be measured against the latest APPROVED revision, not revision 1.
def _rev(po_id, revision, approval, price):
    p = po(po_id, lines=[line(1, price=price)])
    p.revision, p.approval = revision, approval
    return p


def test_bare_po_reference_resolves_to_latest_approved_revision():
    ds = deal(_rev("PO-1", None, None, "10.00"), _rev("PO-1 (Rev 2)", 2, "approved", "12.00"),
              _rev("PO-1 (Rev 3)", 3, "pending", "15.00"), inv(lines=[line(1, price="12.00")]))
    links = link(ds, CFG)
    assert links.invoice_po["INV-1"].doc_id == "PO-1 (Rev 2)"
    (lk,) = links.line_links
    assert lk.po_line.unit_price == D("12.00")


def test_revision_is_read_from_the_id_when_the_column_is_empty():
    ds = deal(po("PO-1"), po("PO-1 (Rev 4)"), inv())
    assert link(ds, CFG).invoice_po["INV-1"].doc_id == "PO-1 (Rev 4)"


def test_no_approved_revision_falls_back_to_the_cited_po():
    ds = deal(_rev("PO-1", 1, "pending", "10.00"), _rev("PO-1 (Rev 2)", 2, "rejected", "12.00"), inv())
    assert link(ds, CFG).invoice_po["INV-1"].doc_id == "PO-1"


def test_an_invoice_citing_a_revision_keeps_it():
    ds = deal(po("PO-1"), po("PO-1 (Rev 2)"), po("PO-1 (Rev 3)"), inv(po_id="PO-1 (Rev 2)"))
    assert link(ds, CFG).invoice_po["INV-1"].doc_id == "PO-1 (Rev 2)"
