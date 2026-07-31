"""Which currency the money on a document is actually in.

A price with no symbol against it is still a price — the number is extracted fine. What can
go wrong is the CURRENCY, and getting that wrong silently rescales every figure downstream:
converted_amount_usd, the spend totals, the value-found headline, the credit note we ask a
supplier for.

Two rules here, both of which end at a human rather than a guess:

  * more than one currency symbol printed -> we do not know which one the totals are in,
    so the document stops for review instead of picking one;
  * a bare "$" is not necessarily USD -> resolve it from evidence on the document (or the
    supplier master) and, when nothing says, stop for review rather than assume.
"""
import pytest

from src.services.extraction.context_layer import (
    currency_conflict, resolve_dollar_currency, DOLLAR_CURRENCIES,
)


# ---- gap 3: more than one symbol -----------------------------------------

def test_two_symbols_on_one_document_is_a_conflict():
    # Freight billed in USD, duty in GBP. Every line inherits ONE header currency, so
    # whichever we pick, some of the money is wrong — and nothing downstream would know.
    text = "Freight charge $4,200.00\nImport duty £860.00\nTotal due £5,060.00"
    conflict = currency_conflict({"currency": "GBP"}, text)
    assert conflict is not None
    assert set(conflict["symbols"]) == {"USD", "GBP"}
    assert conflict["header"] == "GBP"


def test_one_symbol_is_not_a_conflict():
    assert currency_conflict({"currency": "GBP"}, "Total £5,060.00") is None


def test_no_symbol_at_all_is_not_a_conflict():
    # An ISO code with no symbol is the normal shape of a plain-text invoice.
    assert currency_conflict({"currency": "GBP"}, "Currency: GBP\nTotal 5,060.00") is None


def test_a_symbol_the_header_agrees_with_is_not_a_conflict():
    assert currency_conflict({"currency": "USD"}, "Total $4,200.00") is None


def test_a_second_symbol_that_is_only_a_currency_LIST_is_not_a_conflict():
    # Payment blocks routinely print "we accept £/$/€" — that is not the invoice's money.
    text = ("Total due £5,060.00\n"
            "We accept payment in £ / $ / € by bank transfer.")
    assert currency_conflict({"currency": "GBP"}, text) is None


def test_the_conflict_says_what_it_saw():
    text = "Parts $100.00\nLabour €200.00"
    conflict = currency_conflict({"currency": "USD"}, text)
    assert "USD" in conflict["detail"] and "EUR" in conflict["detail"]
    assert conflict["evidence"]


# ---- gap 1: a bare "$" ----------------------------------------------------

def test_a_dollar_document_that_names_its_country_is_resolved_from_it():
    # The schema accepts CAD/AUD/SGD/HKD/NZD, so "$" alone never means USD by itself.
    got = resolve_dollar_currency({"country": "Canada"}, "Total $1,200.00")
    assert got == ("CAD", "the document states country 'Canada'")


def test_a_dollar_document_that_spells_the_code_out_is_resolved_from_that():
    got = resolve_dollar_currency({}, "All amounts in AUD.\nTotal $1,200.00")
    assert got[0] == "AUD"
    assert "AUD" in got[1]


def test_the_printed_code_outranks_the_country():
    # A Canadian entity can invoice in USD and say so. What it SAYS wins.
    got = resolve_dollar_currency({"country": "Canada"}, "Amounts in USD\nTotal $1,200.00")
    assert got[0] == "USD"


def test_a_dollar_document_with_nothing_to_go_on_is_not_resolved():
    # No country, no code, no supplier default: we do not know, and USD is a guess that
    # silently rescales the money. Returns None so the caller routes it to a human.
    assert resolve_dollar_currency({}, "Total $1,200.00") is None


def test_what_people_taught_us_about_the_supplier_is_the_last_resort_not_the_first():
    # supplier_default_currency is populated (dispatch._resolve_bare_dollar_currency_hint)
    # ONLY from a currency several humans corrected this supplier's invoices to — never
    # from proc.bp_supplier.default_currency, which is a static guess about the vendor and
    # would auto-resolve documents that used to stop for review.
    row = {"supplier_default_currency": "SGD"}
    assert resolve_dollar_currency(row, "Total $1,200.00")[0] == "SGD"
    # ...but anything the DOCUMENT says beats even that.
    assert resolve_dollar_currency({**row, "country": "Canada"},
                                   "Total $1,200.00")[0] == "CAD"


def test_every_dollar_currency_the_schema_accepts_is_resolvable():
    # If the schema will store it, this has to be able to reach it — otherwise those
    # documents can only ever be wrong or blocked.
    for code, country in DOLLAR_CURRENCIES.items():
        assert resolve_dollar_currency({"country": country}, "Total $1.00")[0] == code


def test_a_non_dollar_document_is_none_of_this_functions_business():
    assert resolve_dollar_currency({"country": "Canada"}, "Total £1,200.00") is None
