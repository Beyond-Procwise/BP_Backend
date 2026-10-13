"""The selling entity's profile: nothing defaulted, the whole of it replaced on save."""
import pytest

from src.services import entity_profile as ep


def test_a_profile_nobody_has_filled_in_is_all_nulls_not_a_stand_in_company():
    shaped = ep.shape(None)
    assert set(shaped) == set(ep.FIELDS) | {"bank"}
    assert all(shaped[f] is None for f in ep.FIELDS)
    assert shaped["bank"] == {"name": None, "sortCode": None, "accountNumber": None, "iban": None}


def test_only_known_fields_are_kept_and_blanks_are_null():
    cleaned = ep.clean({"legalName": "  Northwind Ltd ", "tradingName": " ", "colour": "red",
                        "bank": {"name": "Bank", "pin": "1234"}})
    assert cleaned["legalName"] == "Northwind Ltd" and cleaned["tradingName"] is None
    assert "colour" not in cleaned and "pin" not in cleaned["bank"]


@pytest.mark.parametrize("profile", [{"legalName": 42}, {"bank": "Barclays"},
                                     {"logoText": "TOOLONG"}, {"vatNumber": "x" * 201}])
def test_a_field_of_the_wrong_kind_or_size_is_refused(profile):
    with pytest.raises(ValueError):
        ep.clean(profile)
