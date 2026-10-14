"""The net total is the one that makes the document's own sums close.

Orbis ORB-Q-6612 (V1) prints a 3-year contract value next to a Year 1 net, Year 1
VAT and Year 1 total, as spreadsheet rows ("label | value" -- no colon, so the
labelled-total reader sees nothing). The model took the 3-year value as the net:
3,371,910 + 223,200 != 1,339,200. V2 and V3 of the same quote were read at Year 1,
so the three rounds of one bid were not comparable.

When the model's set does not close but its VAT and grand total do give a net
that is printed on the page, that printed net is the document's own answer.
"""

from src.services.extraction.context_layer import _reconcile_money

_ORBIS_V1 = """## Sheet: Order Form

| Line item | Year 1 (£) | Year 2 (£) | Year 3 (£) | 3-yr subtotal (£) |
| Platform subscription | 1048000 | 1100400 | 1155420 | 3303820 |
| Implementation & onboarding (one-off, Year 1) | 68000 | 0 | 0 | 68000 |
| Total contract value (TCV), 3 years |  |  |  | 3371910 |
| Year 1 total (ex-VAT) |  |  |  | 1116000 |
| VAT @ 20% |  |  |  | 223200 |
| YEAR 1 TOTAL (incl. VAT) |  |  |  | 1339200 |
"""


def _row(sub, tax, tot):
    return {"total_amount": sub, "tax_amount": tax, "total_amount_incl_tax": tot}


def test_the_printed_net_that_closes_the_sums_replaces_a_contract_value():
    out = _reconcile_money(_row(3371910.0, 223200.0, 1339200.0), _ORBIS_V1, "quote")
    assert out["total_amount"] == 1116000.0
    assert out["tax_amount"] == 223200.0
    assert out["total_amount_incl_tax"] == 1339200.0


def test_a_net_the_page_never_prints_is_not_invented():
    text = _ORBIS_V1.replace("| Year 1 total (ex-VAT) |  |  |  |  | 1116000 |\n", "") \
                    .replace("1116000", "")
    out = _reconcile_money(_row(3371910.0, 223200.0, 1339200.0), text, "quote")
    assert out["total_amount"] == 3371910.0


def test_the_digits_inside_a_longer_number_do_not_count_as_printed():
    text = _ORBIS_V1.replace("1116000", "11160000")
    out = _reconcile_money(_row(3371910.0, 223200.0, 1339200.0), text, "quote")
    assert out["total_amount"] == 3371910.0


def test_figures_that_already_close_are_left_alone():
    out = _reconcile_money(_row(1116000.0, 223200.0, 1339200.0), _ORBIS_V1, "quote")
    assert out["total_amount"] == 1116000.0


def test_a_printed_amount_is_recognised_with_or_without_thousands_separators():
    from src.services.extraction.context_layer import _number_printed
    assert _number_printed(1116000.0, "Year 1 total: £1,116,000.00")
    assert _number_printed(1116000.0, "| 1116000 |")
    assert not _number_printed(1116000.0, "| 1116000.5 |")
    assert not _number_printed(1116000.0, "| 21116000 |")
    assert _number_printed(994.5, "Net 994.50")
