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
