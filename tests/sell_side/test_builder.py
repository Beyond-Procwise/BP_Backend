"""The Quote builder's arithmetic and its text cleaning. No database."""
from decimal import Decimal as D

import pytest

from src.services.sell_side import builder
from src.services.sell_side.money import builder_line, total_builder


def test_a_line_is_net_of_the_discount_the_seller_typed():
    assert builder_line(D(2), D(100), D(10), None).line_total == D("180.00")


def test_a_cost_of_zero_is_not_a_hundred_percent_margin():
    line = builder_line(D(1), D(100), D(0), D(0))
    assert not line.costed and line.line_margin is None and line.line_margin_pct is None


def test_an_uncosted_line_is_left_out_of_the_margin_on_both_sides():
    """60 cost on 100, beside an uncosted 100: the margin is 40%, not 70%."""
    t = total_builder([builder_line(D(1), D(100), D(0), D(60)),
                       builder_line(D(1), D(100), D(0), None)])
    assert t.total_ex_tax == D("200.00")
    assert (t.total_cost, t.total_margin, t.margin_pct) == (D("60.00"), D("40.00"), D("0.4000"))
    assert (t.costed_lines, t.uncosted_lines) == (1, 1)


def test_no_costed_line_means_no_margin_at_all():
    t = total_builder([builder_line(D(3), D(50), D(0), None)])
    assert t.total_ex_tax == D("150.00")
    assert (t.total_cost, t.total_margin, t.margin_pct) == (None, None, None)
    assert (t.costed_lines, t.uncosted_lines) == (0, 1)


def test_the_screens_own_example_adds_up():
    """UI quoteBuilder.contract.test.js: 2 x 100 + 300 = 500 revenue on 220 cost, 56%."""
    t = total_builder([builder_line(D(2), D(100), D(0), D(60)),
                       builder_line(D(1), D(300), D(0), D(100))])
    assert (t.total_ex_tax, t.total_cost, t.margin_pct) == (D("500.00"), D("220.00"), D("0.5600"))


def test_a_below_cost_quote_has_a_negative_margin_not_a_missing_one():
    t = total_builder([builder_line(D(1), D(90), D(0), D(100))])
    assert t.total_margin == D("-10.00")


@pytest.mark.parametrize("dirty, clean", [
    ("<p>Eleven <b>sites</b></p>", "<p>Eleven <b>sites</b></p>"),
    ('<p onclick="x()">hi</p>', "<p>hi</p>"),
    ("<script>alert(1)</script><p>ok</p>", "<p>ok</p>"),
    ('<a href="javascript:x">link</a>', "link"),
    ('<img src=x onerror=alert(1)>after', "after"),
    ("<style>p{}</style>kept", "kept"),
    ("a < b & c", "a &lt; b &amp; c"),
    ("line<br>two<br/>three", "line<br>two<br>three"),
    ("<p>&lt;script&gt;x&lt;/script&gt;</p>", "<p>&lt;script&gt;x&lt;/script&gt;</p>"),
])
def test_rich_text_keeps_a_closed_list_of_tags_and_no_attributes(dirty, clean):
    assert builder.clean_html(dirty, "scopeHtml") == clean


def test_rich_text_has_a_length_limit():
    with pytest.raises(ValueError):
        builder.clean_html("x" * (builder.MAX_HTML + 1), "scopeHtml")


def _draft(**over):
    return {"currency": "GBP", "validUntil": "2099-01-01", "issuedAt": "2026-10-13",
            "lines": [{"tag": "service", "description": "Work", "qty": 1, "unitPrice": 100}],
            **over}


@pytest.mark.parametrize("over, says", [
    ({"lines": [{"qty": 0, "unitPrice": 1}]}, "qty"),
    ({"lines": [{"qty": 1, "unitPrice": -1}]}, "unitPrice"),
    ({"lines": [{"qty": 1}]}, "unitPrice"),
    ({"lines": [{"qty": 1, "unitPrice": 1, "cost": -5}]}, "cost"),
    ({"lines": [{"qty": 1, "unitPrice": 1, "discountPct": 101}]}, "discountPct"),
    ({"lines": [{"qty": 1, "unitPrice": 1, "tag": "gift"}]}, "tag"),
    ({"lines": [{"qty": 1, "unitPrice": "1.23456"}]}, "decimal places"),
    ({"lines": [{"qty": "NaN", "unitPrice": 1}]}, "not a number"),
    ({"validUntil": "2026-01-01"}, "before the quote date"),
    ({"validUntil": None}, "validUntil"),
    ({"currency": "pounds"}, "currency"),
    ({"terms": {"lengthMonths": 1.5}}, "lengthMonths"),
])
def test_a_draft_that_cannot_be_priced_is_refused_with_the_field_named(over, says):
    with pytest.raises(ValueError, match=says):
        builder._clean(_draft(**over))


def test_a_draft_may_be_saved_with_no_lines_yet():
    assert builder._clean(_draft(lines=[]))["lines"] == []


def test_terms_keep_only_the_fields_a_quote_has():
    terms = builder._clean(_draft(terms={"payment": "Net 30", "lengthMonths": 12,
                                         "anything": "else"}))["terms"]
    assert terms == {"payment": "Net 30", "start": None, "lengthMonths": 12,
                     "renewal": None, "sla": None}


class _Cur:
    """A cursor that answers the three questions _check_deal asks, in order."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def execute(self, sql, params=()):
        pass

    def fetchone(self):
        return self.answers.pop(0)


def test_a_deal_that_has_been_invoiced_is_closed_to_a_new_quote():
    with pytest.raises(builder.StateConflict, match="closed"):
        builder._check_deal(_Cur({"?column?": 1}, {"invoices": 2}), "D1", for_quote=None)


def test_a_deal_nobody_has_heard_of_is_not_found():
    with pytest.raises(builder.NotFound):
        builder._check_deal(_Cur(None), "D1", for_quote=None)
