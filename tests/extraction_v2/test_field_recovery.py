"""Tests for deterministic field recovery — no-fabrication contract.

The hard rule under test: if the parsed text does NOT contain explicit
evidence of a field, recovery must leave that field NULL. We never
guess, default, or back-derive from unrelated context.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.field_recovery import (  # noqa: E402
    RecoveryReport, recover_fields,
)


# -- payment_terms --------------------------------------------------------

def test_payment_terms_net_30():
    h = {}
    r = recover_fields(h, parsed_text="Payment Terms: Net 30", doc_type="Invoice")
    assert h["payment_terms"] == "Net 30"
    assert any(f == "payment_terms" for (f, _, _) in r.fields_recovered)


def test_payment_terms_due_on_receipt():
    h = {}
    recover_fields(h, parsed_text="Due on Receipt", doc_type="Invoice")
    assert h["payment_terms"] == "Due on receipt"


def test_payment_terms_payable_within_60():
    h = {}
    recover_fields(h, parsed_text="Payable within 60 days", doc_type="Invoice")
    assert h["payment_terms"] == "Net 60"


def test_payment_terms_not_present_stays_null():
    """No payment terms language anywhere → field stays absent."""
    h = {}
    text = "Invoice 12345 from ACME Co. for £500. Thank you."
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert "payment_terms" not in h


def test_payment_terms_does_not_overwrite():
    h = {"payment_terms": "EOM 15"}
    recover_fields(h, parsed_text="Net 30", doc_type="Invoice")
    assert h["payment_terms"] == "EOM 15"


# -- validity_date / expected_delivery_date --------------------------------

def test_validity_date_until_explicit():
    h = {}
    recover_fields(h, parsed_text="Quote valid until 30 Jun 2026", doc_type="Quote")
    assert h["validity_date"] == "2026-06-30"


def test_validity_date_relative_with_quote_date():
    h = {"quote_date": "2026-05-01"}
    recover_fields(h, parsed_text="Validity: 14 days", doc_type="Quote")
    assert h["validity_date"] == "2026-05-15"


def test_validity_date_relative_without_quote_date_stays_null():
    """No anchor date → can't compute → leave NULL."""
    h = {}
    recover_fields(h, parsed_text="Valid for 14 days", doc_type="Quote")
    assert "validity_date" not in h


def test_expected_delivery_date_extraction():
    h = {}
    recover_fields(h, parsed_text="Expected delivery: 15 May 2026",
                   doc_type="Purchase_Order")
    assert h["expected_delivery_date"] == "2026-05-15"


def test_expected_delivery_date_absent_stays_null():
    h = {}
    recover_fields(h, parsed_text="PO 99999 ACME Co. Quantity 5",
                   doc_type="Purchase_Order")
    assert "expected_delivery_date" not in h


# -- incoterm --------------------------------------------------------------

def test_incoterm_fob():
    h = {}
    recover_fields(h, parsed_text="Incoterm: FOB Southampton",
                   doc_type="Purchase_Order")
    assert h["incoterm"] == "FOB"


def test_incoterm_ddp():
    h = {}
    recover_fields(h, parsed_text="Delivery DDP destination",
                   doc_type="Purchase_Order")
    assert h["incoterm"] == "DDP"


def test_incoterm_absent_stays_null():
    h = {}
    recover_fields(h, parsed_text="PO 1234 widgets quantity 5",
                   doc_type="Purchase_Order")
    assert "incoterm" not in h


# -- currency --------------------------------------------------------------

def test_currency_iso_code_in_text():
    h = {}
    recover_fields(h, parsed_text="Total: GBP 1,000.00", doc_type="Invoice")
    assert h["currency"] == "GBP"


def test_currency_symbol_fallback():
    h = {}
    recover_fields(h, parsed_text="Subtotal £500.00", doc_type="Invoice")
    assert h["currency"] == "GBP"


def test_currency_already_set_not_overwritten():
    h = {"currency": "EUR"}
    recover_fields(h, parsed_text="GBP 500", doc_type="Invoice")
    assert h["currency"] == "EUR"


# -- tax (no fabrication contract) ----------------------------------------

def test_tax_percent_explicit_vat_at():
    h = {}
    recover_fields(h, parsed_text="VAT @ 20% on subtotal", doc_type="Invoice")
    assert h["tax_percent"] == 20.0


def test_tax_percent_with_decimal():
    h = {}
    recover_fields(h, parsed_text="Sales Tax (8.5%):", doc_type="Invoice")
    assert h["tax_percent"] == 8.5


def test_tax_percent_absent_stays_null():
    """No tax language → tax_percent NOT filled. Never derive from
    subtotal + total — that's the closure invariant's job to FLAG, not
    field_recovery's job to GUESS."""
    h = {"subtotal": 100.0, "invoice_total_incl_tax": 120.0}
    recover_fields(h, parsed_text="Subtotal 100. Total 120.", doc_type="Invoice")
    assert "tax_percent" not in h


def test_tax_amount_labelled():
    h = {}
    recover_fields(h, parsed_text="VAT: £24.00", doc_type="Invoice")
    assert h["tax_amount"] == 24.0


def test_tax_amount_absent_stays_null():
    h = {}
    recover_fields(h, parsed_text="Subtotal £100. Total £120.", doc_type="Invoice")
    assert "tax_amount" not in h


def test_tax_amount_does_not_overwrite():
    h = {"tax_amount": 99.0}
    recover_fields(h, parsed_text="VAT: £24.00", doc_type="Invoice")
    assert h["tax_amount"] == 99.0


def test_tax_percent_implausible_rejected():
    """Rate > 30% or 0% is implausible — reject rather than fabricate."""
    h = {}
    recover_fields(h, parsed_text="Tax rate 99%", doc_type="Invoice")
    assert "tax_percent" not in h


# -- country (explicit mention) -------------------------------------------

def test_country_explicit_united_kingdom():
    h = {}
    text = "Bill to: ACME Ltd, 1 High St, London SW1A 1AA, United Kingdom"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert h["country"] == "United Kingdom"


def test_country_explicit_normalises_great_britain():
    h = {}
    text = "Ship to: Great Britain"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert h["country"] == "United Kingdom"


def test_country_postcode_derived_when_no_explicit():
    h = {}
    text = "Bill to: 1 High St, London SW1A 1AA"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert h["country"] == "United Kingdom"


def test_country_absent_stays_null():
    """No address, no postcode, no country mention → NULL."""
    h = {}
    text = "Invoice 12345 widgets quantity 5"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert "country" not in h


# -- region ---------------------------------------------------------------

def test_region_from_uk_postcode_prefix():
    h = {}
    recover_fields(h, parsed_text="Office: London EC1A 1BB",
                   doc_type="Invoice")
    assert h["region"] == "Greater London"


def test_region_unmappable_postcode_stays_null():
    """A valid-shape UK postcode whose prefix isn't in our region map
    must NOT default to a guessed region — leave NULL."""
    h = {}
    # "ZZ1 1AA" — valid shape, no real region. Our table has no ZZ entry.
    recover_fields(h, parsed_text="Address: Somewhere ZE1 0AA",
                   doc_type="Invoice")
    # ZE (Shetland) isn't in our table → stays NULL
    assert "region" not in h


def test_region_no_postcode_stays_null():
    h = {}
    recover_fields(h, parsed_text="Bill to: ACME Inc.", doc_type="Invoice")
    assert "region" not in h


# -- city -----------------------------------------------------------------

def test_city_from_address_line():
    h = {}
    text = "Ship to: 1 Main Road, Horsham RH13 5QH"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_city"] == "Horsham"


def test_city_two_word_town():
    h = {}
    text = "Delivery: 5 The Avenue, Milton Keynes MK1 1AA"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_city"] == "Milton Keynes"


def test_city_with_comma_separator_before_postcode():
    """Real procurement docs sometimes use ',' between city and postcode:
    'Building, Newport, NP20 4EG'. The regex must tolerate both space-
    separated and comma-separated layouts."""
    h = {}
    text = (
        "Ship To:\n"
        "City of Newport Council\n"
        "45 Riverfront Plaza, Civic Centre\n"
        "Building, Newport, NP20 4EG,\n"
        "United Kingdom\n"
    )
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_city"] == "Newport"
    assert h["postal_code"] == "NP20 4EG"


def test_city_with_newline_separator_before_postcode():
    """Layout: city on its own line after a street address line.
    'Suite 14, Innovation House 48 Regent Street\nCambridge, CB2 1FD'
    The newline acts as the city-leading separator."""
    h = {}
    text = (
        "Suite 14, Innovation House 48 Regent Street\n"
        "Cambridge, CB2 1FD United Kingdom\n"
    )
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_city"] == "Cambridge"
    assert h["postal_code"] == "CB2 1FD"


def test_city_no_comma_no_extraction():
    """No comma before the postcode-preceding token → don't grab
    company-name garbage. Conservative."""
    h = {}
    text = "ACME Industries Limited Horsham RH13 5QH"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # No address comma → we refuse to guess
    assert "delivery_city" not in h


def test_city_blocklisted_token_rejected():
    h = {}
    text = "Office, Industrial RH13 5QH"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert "delivery_city" not in h


# -- postcode -------------------------------------------------------------

def test_postcode_uk():
    h = {}
    text = "Address: 1 High St, Horsham RH13 5QH"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "RH13 5QH"


def test_postcode_absent_stays_null():
    h = {}
    text = "ACME Co. invoice"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert "postal_code" not in h


# -- never_overwrites contract --------------------------------------------

def test_never_overwrites_existing_values():
    """Most fields are write-once: never overwrite a present value.

    Exception: ``delivery_region`` / ``region`` IS overwritten when
    postcode-derived evidence contradicts the existing value (covered by
    the dedicated test below). Use a UK postcode whose prefix isn't in
    our region map (so no contradiction is detectable) for this test.
    """
    h = {
        "payment_terms": "Net 45",
        "expected_delivery_date": "2026-12-31",
        "incoterm": "EXW",
        "tax_percent": 12.5,
        "tax_amount": 50.0,
        "currency": "USD",
        "ship_to_country": "Germany",
        "postal_code": "12345",
        "delivery_region": "Some Region",
    }
    # Use ZE1 0AA — a real UK postcode shape but its 'ZE' prefix isn't in
    # our region map, so no contradicting value is computable.
    text = ("VAT @ 20%, Net 30, FOB origin, expected delivery 1 Jan 2027, "
            "VAT: £999.00, GBP, United Kingdom, ZE1 0AA")
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # Every pre-set value preserved exactly
    assert h["payment_terms"] == "Net 45"
    assert h["expected_delivery_date"] == "2026-12-31"
    assert h["incoterm"] == "EXW"
    assert h["tax_percent"] == 12.5
    assert h["tax_amount"] == 50.0
    assert h["currency"] == "USD"
    assert h["ship_to_country"] == "Germany"
    assert h["postal_code"] == "12345"
    assert h["delivery_region"] == "Some Region"


def test_region_overrides_when_postcode_contradicts():
    """If the LLM wrote a region that contradicts the postcode-derived
    region, recovery overrides it — postcode is the stronger signal
    because it's paired with the same address."""
    h = {"delivery_region": "West Sussex"}
    # M1 7HY → Manchester area, our map says 'Greater Manchester'
    text = "Ship to: 1 High St, Manchester M1 7HY"
    r = recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_region"] == "Greater Manchester"
    sources = {f: s for (f, _, s) in r.fields_recovered}
    assert sources["delivery_region"] == "postcode.region.contradicts"


def test_region_does_not_override_when_postcode_agrees():
    """No-op when the LLM's region matches the postcode-derived one.
    Avoids spurious provenance entries."""
    h = {"delivery_region": "West Sussex"}
    text = "Ship to: 1 High St, Horsham RH13 5QH"  # RH → West Sussex
    r = recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_region"] == "West Sussex"
    # No region recovery should be reported
    assert not any(f in ("region", "delivery_region") for (f, _, _) in r.fields_recovered)


def test_multiple_postcodes_no_anchor_leaves_null():
    """Two distinct UK postcodes in text without a delivery anchor →
    we can't tell which is the delivery postcode → leave NULL. Country
    is still safe to fill ('United Kingdom')."""
    h = {}
    text = (
        "Supplier: ACME Ltd, Manchester M1 7HY\n"
        "Buyer: Beta Corp, Horsham RH13 5QH\n"
    )
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # postal_code stays NULL — ambiguous
    assert "postal_code" not in h or h.get("postal_code") is None
    # delivery_region stays NULL — can't derive without postcode
    assert "delivery_region" not in h or h.get("delivery_region") is None
    # ship_to_country IS safe to fill — both postcodes are UK
    assert h.get("ship_to_country") == "United Kingdom"


def test_multiple_postcodes_with_anchor_picks_delivery_block():
    """Two postcodes BUT a delivery anchor scopes the search to one
    block → use that block's postcode, no ambiguity."""
    h = {}
    text = (
        "Bill From: ACME, Manchester M1 7HY\n"
        "Ship To:\n"
        "Beta Corp, 1 High St, Horsham RH13 5QH\n"
    )
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "RH13 5QH"
    assert h["delivery_region"] == "West Sussex"


def test_invoice_date_recovery():
    h = {}
    text = "INVOICE\nDate Issued: 27/09/2024\nInvoice No: 148769"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert h.get("invoice_date") == "2024-09-27"


def test_quote_date_recovery():
    h = {}
    text = "QUOTE\nQuote Date: 22 Dec 2024\nQuote No: 128300"
    recover_fields(h, parsed_text=text, doc_type="Quote")
    assert h.get("quote_date") == "2024-12-22"


def test_order_date_recovery_for_po():
    h = {}
    text = "PURCHASE ORDER\nPO Date: 08 Jul 2019\nPO Number: 502004"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h.get("order_date") == "2019-07-08"


def test_due_date_recovery():
    h = {}
    text = "Invoice\nDue Date: 27/12/2024\nThanks for your business"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    assert h.get("due_date") == "2024-12-27"


def test_date_recovery_does_not_overwrite_existing():
    h = {"invoice_date": "2025-01-01"}
    recover_fields(h, parsed_text="Date Issued: 27/09/2024", doc_type="Invoice")
    assert h["invoice_date"] == "2025-01-01"  # preserved


def test_date_recovery_skipped_when_no_evidence():
    h = {}
    recover_fields(h, parsed_text="Some random text. No dates here.",
                   doc_type="Invoice")
    assert "invoice_date" not in h or h.get("invoice_date") is None


def test_postcode_existing_value_not_overwritten():
    """When the LLM has already provided a postal_code, recovery must
    not overwrite it — even if the parsed text contains a different
    postcode that appears earlier. Procurement docs vary: sometimes the
    supplier's letterhead is at the top, sometimes the buyer's. We
    can't reliably guess from position alone, so we trust the LLM's
    value."""
    h = {"postal_code": "RH13 5QH"}
    text = (
        "85 Brook Street\n"
        "Manchester, M1 7HY\n"           # different postcode FIRST
        "DESCRIPTION\n"
        "10 Redkiln Way Horsham West Sussex RH13 5QH\n"  # LLM's choice — LATER
    )
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # LLM's postcode preserved — recovery no longer second-guesses it
    assert h["postal_code"] == "RH13 5QH"
    # Region derives from header.postal_code (RH13 → West Sussex)
    assert h["delivery_region"] == "West Sussex"


def test_region_derived_from_existing_postal_code_not_text_scan():
    """Critical consistency invariant: when the header already has a
    postal_code, region must be derived from THAT postcode, not from a
    fresh text scan that could pick up an unrelated postcode in the doc.

    Repro: PO has postal_code='RH13 5QH' (West Sussex) already set, but
    parsed_text contains 'B15 2TH' (West Midlands) elsewhere. Recovery
    must override delivery_region using RH13's region (West Sussex), not
    B15's region (West Midlands)."""
    h = {"postal_code": "RH13 5QH", "delivery_region": "Some Other Region"}
    # Text contains a Birmingham postcode that the regex would otherwise hit
    text = "Bill From: 1 Trinity Sq, Birmingham B15 2TH\nShip To: Horsham"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["delivery_region"] == "West Sussex", (
        "Region must derive from header.postal_code (RH13→West Sussex), "
        "NOT from a text scan that finds the Birmingham postcode"
    )
    assert h["postal_code"] == "RH13 5QH"  # untouched


def test_empty_text_returns_empty_report():
    h = {}
    r = recover_fields(h, parsed_text="", doc_type="Invoice")
    assert isinstance(r, RecoveryReport)
    assert r.fields_recovered == []
    assert h == {}


def test_unknown_doc_type_returns_empty_report():
    h = {}
    r = recover_fields(h, parsed_text="Net 30 FOB", doc_type="Mystery")
    assert r.fields_recovered == []
    assert h == {}


# -- address-block-aware extraction (mixed-source guard) -----------------

_TWO_ADDRESS_TEXT = (
    "PURCHASE ORDER\n"
    "Bill From:\n"
    "InfoTech Consulting Ltd\n"
    "10 Redkiln Way, Horsham, West Sussex\n"
    "RH13 5QH, United Kingdom\n"
    "\n"
    "Ship To:\n"
    "Acme Manufacturing\n"
    "100 Industrial Road, Manchester\n"
    "M1 7HY, United Kingdom\n"
    "\n"
    "Item Description Qty Unit Price Total\n"
    "Widget A 5 200.00 1000.00\n"
)


def test_postcode_picks_delivery_block_not_supplier():
    """Doc has both supplier (RH13 5QH) and ship-to (M1 7HY).
    Recovery must pick M1 7HY because that's in the Ship To block."""
    h = {}
    recover_fields(h, parsed_text=_TWO_ADDRESS_TEXT, doc_type="Purchase_Order")
    assert h["postal_code"] == "M1 7HY"


def test_region_derived_from_delivery_postcode_not_supplier():
    h = {}
    recover_fields(h, parsed_text=_TWO_ADDRESS_TEXT, doc_type="Purchase_Order")
    # M → Greater Manchester, NOT West Sussex
    assert h["delivery_region"] == "Greater Manchester"


def test_city_picks_delivery_block_not_supplier():
    h = {}
    recover_fields(h, parsed_text=_TWO_ADDRESS_TEXT, doc_type="Purchase_Order")
    assert h["delivery_city"] == "Manchester"


def test_anchor_variants():
    """Test 'Delivery address:' anchor variant."""
    text = (
        "From: SupplierCo, RH13 5QH\n"
        "Delivery address: 5 Mill Lane, Cambridge CB2 1FD\n"
    )
    h = {}
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "CB2 1FD"
    assert h["delivery_region"] == "Cambridgeshire"


def test_anchor_deliver_to():
    text = (
        "Issued by: ACME, RH13 5QH\n"
        "Deliver to:\n"
        "Beta Corp, 1 Way St, Newport NP20 4EG\n"
    )
    h = {}
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "NP20 4EG"
    assert h["delivery_region"] == "Newport"


def test_no_anchor_falls_back_to_global_first_match():
    """When no anchor exists, fall back to the global first match.
    Single-address invoices typically have no Ship To block."""
    h = {}
    text = "Invoice from ACME at 1 Main St, Horsham RH13 5QH"
    recover_fields(h, parsed_text=text, doc_type="Invoice")
    # No anchor → first global postcode wins
    assert h.get("country") == "United Kingdom"
    assert h.get("region") == "West Sussex"


def test_terminator_cuts_block_before_bill_to():
    """Make sure the Bill To block doesn't leak into Ship To."""
    text = (
        "Ship To: Beta Corp, Manchester M1 7HY\n"
        "Bill To: ACME, Horsham RH13 5QH\n"
    )
    h = {}
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # Must pick M1 (Ship To), not RH13 (Bill To)
    assert h["postal_code"] == "M1 7HY"
    assert h["delivery_region"] == "Greater Manchester"


def test_anchor_with_no_postcode_inside_falls_back():
    """Anchor present but its block has no postcode → fall back to
    global first match (better than NULL — there IS evidence in the
    document)."""
    text = (
        "Ship To: TBD\n"
        "Supplier: ACME, Horsham RH13 5QH\n"
    )
    h = {}
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # Falls back to the only postcode in the doc
    assert h["postal_code"] == "RH13 5QH"


def test_report_records_source_for_each_recovery():
    h = {}
    text = ("Payment Terms: Net 30. VAT @ 20%. VAT: £24.00. "
            "Ship to: 1 High St, Horsham RH13 5QH, United Kingdom. FOB.")
    r = recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    sources = {f: s for (f, _, s) in r.fields_recovered}
    assert sources.get("payment_terms") == "regex.payment_terms"
    assert sources.get("tax_percent") == "regex.tax_percent"
    assert sources.get("tax_amount") == "regex.tax_amount"
    assert sources.get("incoterm") == "regex.incoterm"
    assert sources.get("postal_code") == "regex.uk_postcode"
    assert sources.get("ship_to_country") == "regex.country_explicit"
