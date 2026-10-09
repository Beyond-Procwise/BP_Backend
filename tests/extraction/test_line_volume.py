"""The volume a line is for, read from its description (line_volume.py).

DEALV3-202: Fortis quoted "Service Desk (24x7, 2,400 users)" at £540,000, the others "3,000
users" at £584,000 / £610,000. Billed as quantity 1 year each, Fortis looked cheapest; per user
it is the dearest. The descriptions below are the corpus's own.
"""
from decimal import Decimal

from src.services.extraction.line_volume import add_line_volumes, volume_from_description


def test_reads_the_volume_a_service_line_is_for():
    assert volume_from_description("Service Desk (24x7, 2,400 users) — annual") == (Decimal("2400"), "users")
    assert volume_from_description("Endpoint management (3,000 seats) — annual") == (Decimal("3000"), "seats")
    assert volume_from_description("Platform licence — Enterprise (240 seats)") == (Decimal("240"), "seats")
    assert volume_from_description("Microsoft 365 E3 — 1 licence") == (Decimal("1"), "licences")


def test_never_reads_a_product_name_size_or_range_as_a_volume():
    for d in ["Compact 24–27 Unit ITM001020", "Professional 13–14 Unit ITM000831", "Premium A4/A3 Unit ITM001051",
              "Service Desk (24x7)", "On-call / out-of-hours engineering", "Training for 13–14 users",
              "ITM-240 seats", "", None]:
        assert volume_from_description(d) == (None, None), d


def test_a_description_naming_two_volumes_is_ambiguous():
    assert volume_from_description("Licences: 200 users now, 300 users from year 2") == (None, None)
    # The same volume said twice is still one volume.
    assert volume_from_description("3,000 seats (3,000 seats)") == (Decimal("3000"), "seats")


def test_lines_get_volume_fields_and_keep_their_billing_quantity():
    lines = add_line_volumes([
        {"item_description": "Service Desk (24x7, 2,400 users) — annual", "quantity": 1, "unit_of_measure": "year"},
        {"item_description": "On-call / out-of-hours engineering", "quantity": 52, "unit_of_measure": "week"},
    ])
    assert lines[0] == {"item_description": "Service Desk (24x7, 2,400 users) — annual", "quantity": 1,
                        "unit_of_measure": "year", "volume": Decimal("2400"), "volume_unit": "users"}
    assert "volume" not in lines[1]


def test_a_line_that_already_carries_a_volume_keeps_it():
    lines = add_line_volumes([{"item_description": "Service Desk (2,400 users)", "volume": 2500, "volume_unit": "users"}])
    assert lines[0]["volume"] == 2500
