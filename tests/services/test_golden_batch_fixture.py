from tests.fixtures.deal_clustering import golden_batch as gb
from src.services.version_collapse import collapse_versions

def test_batch_has_28_quotes_that_collapse_to_12_bids():
    qs = gb.quotes()
    assert len(qs) == 28
    assert len(collapse_versions(qs)) == 12

def test_five_pos_each_distinct_supplier():
    pos = gb.purchase_orders()
    assert len(pos) == 5
    assert len({p["po_id"] for p in pos}) == 5

def test_freight_event_has_three_suppliers_incl_null_supplier_swift():
    freight = [q for q in gb.quotes() if q["quote_id"].startswith(("SDP-Q", "CL-2024", "MFS-Q"))]
    sups = {q.get("supplier_id") for q in freight}
    assert None in sups                      # SDP-Q-44120 (Swift) supplier_id is null
    assert len([s for s in sups if s]) == 2  # Condor + Meridian Freight resolved

def test_negative_control_gomez_name_mismatch_is_present():
    q = next(q for q in gb.negative_control_quotes() if q["quote_id"] == "104683")
    po = next(p for p in gb.negative_control_pos() if p["po_id"] == "PO-104683")
    assert q["supplier_id"] == "SUP-GomezGoodAndCross"
    assert po["supplier_name"] == "Gomez, Good and Cross Trading Ltd"
    assert po.get("supplier_id") is None     # forces continuity-scored award detection
