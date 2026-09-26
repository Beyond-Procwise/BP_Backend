from src.services.truth.derived import DERIVED_FIELDS, is_derived


def test_a_field_the_pipeline_computes_is_derived():
    assert is_derived("converted_amount_usd")
    assert is_derived("exchange_rate_to_usd")


def test_a_field_copied_from_the_page_is_not_derived():
    assert not is_derived("invoice_id")
    assert not is_derived("supplier_id")
    assert not is_derived("invoice_amount")


def test_unknown_fields_are_not_derived():
    # Defaulting to derived would excuse every new field from measurement.
    assert not is_derived("some_field_added_next_year")


def test_the_list_is_explicit_and_reviewable():
    assert isinstance(DERIVED_FIELDS, frozenset)
    assert DERIVED_FIELDS, "an empty list means nothing is excused -- state it deliberately"
