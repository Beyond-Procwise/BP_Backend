"""The backfill must harvest only what is genuinely in calculation_details.

Measured shapes on bp_testdb (bp_sqldb has zero opportunity rows):

    Duplicate Invoice Recovery  300  currency, amount_gbp, amount_native,
                                     duplicate_of, band, signals, payment_note,
                                     payment_confirmed, relationship_score
    Invoice Overbilling           6  deal_id, po_total, quote_total,
                                     invoice_total, variance_pct, auto_detect
    Price Benchmark Variance      2  quantity, actual_price, benchmark_price,
                                     item_reference, item_description,
                                     variance_pct, flow_coverage,
                                     risk_score_normalised, auto_detect

So the honest harvest is currency and amount_native for 300 rows, quantity and
unit_price for 2, and nothing structured for the other 6. Everything else is
INDETERMINATE, not inferred.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.backfill_opportunity_structured import harvest  # noqa: E402

D = Decimal

DUPLICATE = {
    "currency": "USD", "amount_gbp": 1200.0, "amount_native": 1500.0,
    "duplicate_of": "INV-9", "band": "high", "signals": ["exact_amount"],
    "payment_note": "paid", "payment_confirmed": True, "relationship_score": 0.9,
}

OVERBILLING = {
    "deal_id": "D-1", "po_total": 1000.0, "quote_total": 900.0,
    "invoice_total": 1400.0, "variance_pct": 40.0, "auto_detect": True,
}

BENCHMARK = {
    "quantity": 10, "actual_price": 55.5, "benchmark_price": 44.0,
    "item_reference": "ITEM-1", "item_description": "Laptop",
    "variance_pct": 26.1, "flow_coverage": 0.8, "risk_score_normalised": 0.5,
    "auto_detect": True,
}


def test_duplicate_invoice_yields_currency_and_native_amount():
    h = harvest(DUPLICATE)
    assert h.currency == "USD"
    assert h.amount_native == D("1500.0")
    assert h.facts_state == "RESOLVED"


def test_the_gbp_amount_is_not_harvested_as_the_native_one():
    """amount_gbp is already converted. Writing it into amount_native would
    restate a converted figure as the document's own, and nothing downstream
    could tell the difference."""
    h = harvest(DUPLICATE)
    assert h.amount_native != D("1200.0")


def test_overbilling_has_nothing_structured_and_says_so():
    """po_total, quote_total and invoice_total are three different documents'
    totals. None of them is 'the' amount of the finding, and picking one would
    be a guess dressed as a harvest."""
    h = harvest(OVERBILLING)
    assert h.facts_state == "INDETERMINATE"
    assert h.currency is None
    assert h.amount_native is None
    assert h.unit_price is None
    assert h.quantity is None
    assert h.uom is None
    assert h.fx_rate is None


def test_benchmark_variance_yields_quantity_and_unit_price():
    h = harvest(BENCHMARK)
    assert h.quantity == D("10")
    assert h.unit_price == D("55.5")
    assert h.facts_state == "RESOLVED"


def test_the_benchmark_price_is_not_harvested_as_the_actual_price():
    """benchmark_price is the comparator, not what was paid."""
    h = harvest(BENCHMARK)
    assert h.unit_price != D("44.0")


def test_a_currency_absent_from_the_payload_stays_null():
    h = harvest(BENCHMARK)
    assert h.currency is None, "no currency key means no currency, not GBP"


@pytest.mark.parametrize("payload", [None, {}, {"note": "n/a"}, {"band": "high"}])
def test_a_row_with_no_parseable_key_is_indeterminate_with_every_column_null(payload):
    h = harvest(payload)
    assert h.facts_state == "INDETERMINATE"
    assert (h.currency, h.amount_native, h.unit_price, h.quantity,
            h.uom, h.uom_normalised, h.fx_rate, h.fx_rate_date) == (None,) * 8


def test_indeterminate_is_not_the_same_as_zero():
    """A finding whose amount could not be resolved must not read as a finding
    worth nothing."""
    h = harvest(OVERBILLING)
    assert h.amount_native is None
    assert h.amount_native != D("0")


def test_money_is_decimal_not_float():
    h = harvest(DUPLICATE)
    assert isinstance(h.amount_native, Decimal)


def test_an_unparseable_number_is_refused_not_coerced():
    h = harvest({"currency": "GBP", "amount_native": "not a number"})
    assert h.amount_native is None
    assert any("UNPARSEABLE" in c for c in h.reason_codes)


def test_a_uom_present_in_the_payload_is_normalised_through_the_same_rules():
    h = harvest({"quantity": 3, "actual_price": 10, "uom": "EACH"})
    assert h.uom == "EACH"
    assert h.uom_normalised == "each"


def test_a_junk_uom_is_carried_raw_and_flagged():
    h = harvest({"quantity": 3, "actual_price": 10, "uom": "30 days from invoice"})
    assert h.uom == "30 days from invoice"
    assert h.uom_normalised is None
    assert "UOM_UNMAPPED" in h.reason_codes
