import datetime as dt
from src.services import deal_assignment_service as das


def test_basename_match_ignores_directory_and_case():
    assert das.basename_match("documents/po/DUNCAN PO526702.pdf", "/tmp/x/duncan po526702.PDF")
    assert not das.basename_match("a/INV1.pdf", "a/INV2.pdf")


def test_mint_document_id_is_deterministic_and_typed():
    a = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    b = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    assert a == b == "DEAL_A2026052891::invoice::INV610366"


def test_resolve_deal_date_prefers_po_expected_delivery():
    po = {"expected_delivery_date": dt.date(2024, 10, 9)}
    assert das.resolve_deal_date(po, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 10, 9)


def test_resolve_deal_date_falls_back_to_invoice_line_then_none():
    assert das.resolve_deal_date(None, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 11, 1)
    assert das.resolve_deal_date(None, inv_line_delivery=None) is None


def test_lookback_deal_identity_from_canonical_po():
    assert das.lookback_deal_id("526702") == "DEALV2-526702"
    assert das.lookback_deal_name("Duncan LLC", "526702") == "Duncan LLC — PO 526702"
