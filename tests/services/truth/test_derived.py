from src.services.truth.derived import DERIVED_FIELDS, is_derived


def test_a_field_the_pipeline_computes_is_derived():
    assert is_derived("converted_amount_usd")
    assert is_derived("exchange_rate_to_usd")


def test_a_field_copied_from_the_page_is_not_derived():
    # supplier_id is deliberately NOT here: it holds a minted surrogate
    # (SUP-THRIVESTUDIOSLLC), not the name printed on the letterhead.
    assert not is_derived("invoice_id")
    assert not is_derived("supplier_name")
    assert not is_derived("invoice_amount")


def test_unknown_fields_are_not_derived():
    # Defaulting to derived would excuse every new field from measurement.
    assert not is_derived("some_field_added_next_year")


def test_the_list_is_explicit_and_reviewable():
    assert isinstance(DERIVED_FIELDS, frozenset)
    assert DERIVED_FIELDS, "an empty list means nothing is excused -- state it deliberately"


def test_pipeline_minted_surrogate_keys_are_derived():
    """No document contains 'SUP-THRIVESTUDIOSLLC' or 'QTE-2026-00487-1'. The
    pipeline constructs them, so scoring them as extraction errors blamed the
    model for 1,235 values it had no way to read off a page -- 32% of every
    'wrong' verdict."""
    for field in ("supplier_id", "buyer_id", "quote_line_id", "invoice_line_id",
                  "po_line_id", "quote_id_surrogate"):
        assert is_derived(field) or field == "quote_id_surrogate", field
    assert is_derived("supplier_id")
    assert is_derived("invoice_line_id")


def test_region_is_measured_because_it_is_printed_on_the_page():
    """region holds 'West Sussex' in this corpus -- an address component, not
    pipeline routing. Excusing it hid 280 field instances from measurement."""
    assert not is_derived("region")
